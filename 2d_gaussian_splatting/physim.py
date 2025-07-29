#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm, trange
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json
import cv2
from PIL import Image
import torchvision.transforms as T
import numpy as np
from submodules.mpm_engine.mpm_solver import MPMSolver
import taichi as ti
from plyfile import PlyData, PlyElement
from utils.graphics_utils import fov2focal
from utils.mesh_utils import GaussianExtractor,GaussianExtractor_sim, to_cam_open3d, post_process_mesh
from time import time as clocktime
import json
import open3d as o3d

def apply_colormap(gray, minmax=None, cmap=None):
    if type(gray) is not np.ndarray:
        gray = gray.detach().cpu().numpy().astype(np.float32)
    gray = gray.squeeze()
    assert len(gray.shape) == 2
    x = np.nan_to_num(gray)  # change nan to 0
    if minmax is None:
        mi = np.min(x)  # get minimum positive value
        ma = np.max(x)
    else:
        mi, ma = minmax
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = np.clip(x,0,1)
    x = (255 * x).astype(np.uint8)
    if cmap == None:
        x_= x
    else:
        x_ = cv2.applyColorMap(x, cmap)
    return x_    




# def physim(dataset : ModelParams, iteration : int, pipeline : PipelineParams, obj_index: int, time : int, sim_obj : int):
#     with torch.no_grad():
#         gaussians = GaussianModel(dataset.sh_degree)
#         scene = Scene(dataset, gaussians, time, load_iteration=iteration, shuffle=False)


#         if sim_obj!=0:
#             ti.init(arch=ti.cuda, device_memory_fraction=0.5, kernel_profiler=True)
#             mpm = MPMSolver(res=(48, 48, 48), size=1, max_num_particles=2 ** 21,
#                     E_scale=0.05, poisson_ratio=0.4, unbounded=True)
#             mpm.set_gravity((0, 0, -4.9))
#             obj_idxs = []

#             for idx in sim_obj:
#                 # points[colors==[idx,idx,idx]]



#                 binarized_voxel, center_xyz, scale_xyz = gaussians.extract_fields(obj_index=idx, resolution=64,binarize_threshold=0.1)
#                 voxel_res = binarized_voxel.shape[0]
#                 pts_on_disk_n3 = np.mgrid[0:voxel_res, 0:voxel_res, 0:voxel_res].reshape(3, -1).T
#                 pts_on_disk_n3 = pts_on_disk_n3[binarized_voxel.flatten() == 1]
#                 pts_on_disk_n3 = pts_on_disk_n3 / (voxel_res // 2) - 1
#                 pts_on_disk_n3 = (pts_on_disk_n3 / scale_xyz) + center_xyz



#                 real_gaussian_particle =gaussians.get_xyz_each_obj_index(idx).detach().cpu().numpy()
#                 real_gaussian_particle_size = real_gaussian_particle.shape[0]
#                 rigid_idx = np.zeros(real_gaussian_particle_size, dtype=bool)
#                 all_particles = np.concatenate([real_gaussian_particle, pts_on_disk_n3], axis=0)
#                 particles, rigid_flag = gaussians.infill_particles(infilling_voxel_res=64, support_per_particles=20, 
#                                                                     real_gaussian_particle=real_gaussian_particle, rigid_idx=rigid_idx, surface_particles=pts_on_disk_n3, particles=all_particles)
            
#                 particles = particles.astype('float32')

#                 mpm.add_particles(particles=particles,
#                             material=MPMSolver.material_elastic,
#                             color=0xFFFF00, motion_override_flag_arr=rigid_flag)
#                 obj_idx = np.zeros(particles.shape[0], dtype=bool)
#                 obj_idx[:real_gaussian_particle_size] = True
#                 obj_idxs.extend(obj_idx)
                    
#             mpm.add_surface_collider(point=(0.0, 0.0, 0.0),
#                                         normal=(0, 0, 1),
#                                         surface=mpm.surface_sticky)
            
            
#             particles_trajectory = []
#             for frame in trange(100):
#                 particles_info = mpm.particle_info()
#                 # real_gaussian_pos = particles_info['position'][:real_gaussian_particle_size]
#                 real_gaussian_pos = particles_info['position'][obj_idxs,:]
#                 particles_trajectory.append(real_gaussian_pos.copy())
#                 override_velocity = [0, 0, 0]
#                 mpm.step(4e-3, override_velocity=override_velocity)
#             particles_trajectory_tn3 = np.stack(particles_trajectory)
#             obj_pc_after = particles_trajectory_tn3[-1]
#             gaussians.save_ply_full(obj_pc_after, sim_obj, os.path.join(args.model_path, f"point_cloud_{time:02d}_before_align.ply")) 

#             bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#             background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#             bg = background
#             render_path = os.path.join(scene.model_path, 'test', "ours_{}_before_align".format(iteration), "renders")
#             gts_path = os.path.join(scene.model_path, 'test', "ours_{}_before_align".format(iteration), "masked")
#             os.makedirs(render_path, exist_ok=True)
#             os.makedirs(gts_path, exist_ok=True)
#             for idx, view in enumerate(tqdm(scene.getTrainCameras(), desc="Rendering progress")):
#                 image_name = view.image_name             
#                 render_pkg = render(view, gaussians, pipeline, bg)
#                 obj_index_image_before = render_pkg["obj_index"]
#                 depth = render_pkg["surf_depth"]
#                 gt_obj_one_hot_image = view.original_obj_one_hot_image.cuda() 
#                 FoVx = view.FoVx
#                 # projection_matrix = view.projection_matrix
#                 # breakpoint()
#                 focal = fov2focal(FoVx, view.image_width)
#                 rendering_before = render_pkg["render"]
#                 mask_before = render_pkg["seg"]
#                 torchvision.utils.save_image(rendering_before, os.path.join(render_path, image_name + ".png"))
#                 torchvision.utils.save_image(mask_before, os.path.join(gts_path, image_name + ".png"))

#                 # breakpoint()

#             C, H, W = obj_index_image_before.shape
#             point_idx = 0
#             xyz = gaussians._xyz.detach().cpu().numpy()
#             _obj_index = gaussians.get_obj_index

#             yy, xx = torch.meshgrid(torch.arange(H, device=obj_index_image_before.device),
#                                     torch.arange(W, device=obj_index_image_before.device),
#                                     indexing='ij') 
#             for c in (sim_obj):
#                 mask = (obj_index_image_before[c] > 0.8)
#                 mask_= (gt_obj_one_hot_image[c] > 0.8)
#                 mask_f = mask.float()
#                 y_center = (yy * mask_f).sum() / mask_f.sum()
#                 x_center = (xx * mask_f).sum() / mask_f.sum()
#                 mask_f_ = mask_.float()
#                 y_center_ = (yy * mask_f_).sum() / mask_f_.sum()
#                 x_center_ = (xx * mask_f_).sum() / mask_f_.sum()
#                 target_depth = depth[int(y_center.item()),int(x_center.item())]
#                 mask = (_obj_index[:,c]>0.50) # *(gaussians._xyz[:,2]>0)
#                 mask = mask.detach().cpu().numpy()
#                 obj_pc_after[point_idx:point_idx+sum(mask)][:,0] = obj_pc_after[point_idx:point_idx+sum(mask)][:,0] -((y_center_-y_center)*target_depth/focal).item()
#                 obj_pc_after[point_idx:point_idx+sum(mask)][:,1] = obj_pc_after[point_idx:point_idx+sum(mask)][:,1] -((x_center_-x_center)*target_depth/focal).item()
#                 point_idx+=sum(mask)


#             gaussians.save_ply_full(obj_pc_after, sim_obj, os.path.join(args.model_path, f"point_cloud_{time:02d}.ply")) 
#         else:
#             gaussians.save_ply(os.path.join(args.model_path, f"point_cloud_{time:02d}.ply"))

       




def physim_pp(dataset : ModelParams, iteration : int, pipeline : PipelineParams, time : int):
    obj_idxs = []
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, time, load_iteration=iteration, shuffle=False)
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(0)
    gaussians.visible_obj_index = viewpoint_cam.visible_obj_index
    real_gaussian_particle_size_list = []
    bg_color = [1,1,1]
    # gaussExtractor = GaussianExtractor_sim(gaussians, render, pipeline, bg_color=bg_color) 
    gaussExtractor = GaussianExtractor(gaussians, render, pipeline, bg_color=bg_color) 
    # gaussians.load_ply_remove_obj(os.path.join(args.model_path,
    #                                     "point_cloud",
    #                                     "iteration_" + str(iteration),
    #                                     "point_cloud.ply"), [2,3,4,5,6])

    gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(scene.getTestCameras())
    depth_trunc = (gaussExtractor.radius * 2.0)
    voxel_size = (depth_trunc / 256)
    sdf_trunc = 5.0 * voxel_size
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    mesh_post = post_process_mesh(mesh, cluster_to_keep=50)
    # mesh_path = os.path.join(scene.model_path, 'test', "ours_{}_before_align".format(iteration))
    # o3d.io.write_triangle_mesh(os.path.join(mesh_path,"mesh.ply"), mesh_post)
    # breakpoint()
    points = np.asarray(mesh_post.vertices)
    colors = np.asarray(mesh_post.vertex_colors)
    del mesh
    del mesh_post
    ti.init(arch=ti.cuda, device_memory_fraction=0.3, kernel_profiler=True)
    mpm = MPMSolver(res=(48, 48, 48), size=1, max_num_particles=2 ** 21,
    E_scale=0.05, poisson_ratio=0.4, unbounded=True)
    sort_index = gaussians.visible_obj_index



    with torch.no_grad():
        start_mpm_time = clocktime()
        for i in sort_index:
            # breakpoint()
            mask = np.all(colors == np.array([i/255, i/255, i/255]), axis=1)
            points_of_idx = points[mask]

            # binarized_voxel, center_xyz, scale_xyz = gaussians.extract_fields(obj_index=i, resolution=64,binarize_threshold=0.1)
            # voxel_res = binarized_voxel.shape[0]
            # pts_on_disk_n3 = np.mgrid[0:voxel_res, 0:voxel_res, 0:voxel_res].reshape(3, -1).T
            # pts_on_disk_n3 = pts_on_disk_n3[binarized_voxel.flatten() == 1]
            # pts_on_disk_n3 = pts_on_disk_n3 / (voxel_res // 2) - 1
            # pts_on_disk_n3 = (pts_on_disk_n3 / scale_xyz) + center_xyz
            real_gaussian_particle =gaussians.get_xyz_each_obj_index(i).detach().cpu().numpy()
            real_gaussian_particle_size = real_gaussian_particle.shape[0]
            real_gaussian_particle_size_list.append(real_gaussian_particle_size)
            rigid_idx = np.zeros(real_gaussian_particle_size, dtype=bool)
            all_particles = np.concatenate([real_gaussian_particle, points_of_idx], axis=0)

            # particles, rigid_flag = gaussians.infill_particles(infilling_voxel_res=64, support_per_particles=20, 
            #                                                     real_gaussian_particle=real_gaussian_particle, rigid_idx=rigid_idx, surface_particles=pts_on_disk_n3, particles=all_particles)
            particles = all_particles.astype('float32')
            # particles = real_gaussian_particle.astype('float32')

            # rigid_flag = np.zeros(particles.shape[0], dtype=bool)
            # particles = real_gaussian_particle
            rigid_flag = rigid_idx
            mpm.add_particles(particles=particles,
                        material=MPMSolver.material_elastic,
                        color=0xFFFF00, motion_override_flag_arr=rigid_flag)
                        
            mpm.add_surface_collider(point=(0.0, 0.0, 0.0),
                                    normal=(0, 0, 1),
                                    surface=mpm.surface_sticky)
            obj_idx = np.zeros(particles.shape[0], dtype=bool)
            obj_idx[:real_gaussian_particle_size] = True
            obj_idxs.extend(obj_idx)
 
        mpm.set_gravity((0, 0, -4.9))
        obj_idxs = np.array(obj_idxs)

        for frame in trange(100):
            particles_info = mpm.particle_info()
            real_gaussian_pos = particles_info['position'][obj_idxs,:]
            override_velocity = [0, 0, 0]
            mpm.step(4e-3, override_velocity=override_velocity)
        end_mpm_time = clocktime()
        print(f"MPM Time : {round(end_mpm_time - start_mpm_time, 3)}")
        
        obj_pc_after = real_gaussian_pos
        gaussians.save_ply_full(obj_pc_after, sort_index, os.path.join(args.model_path, f"point_cloud_{int(time+1):02d}_before_align.ply")) 

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        bg = background
        render_path = os.path.join(scene.model_path, 'test', "ours_{}_before_align".format(iteration), "renders")
        gts_path = os.path.join(scene.model_path, 'test', "ours_{}_before_align".format(iteration), "masked")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)

        for idx, view in enumerate(tqdm(scene.getTrainCameras(), desc="Rendering progress")):
            image_name = view.image_name             
            render_pkg = render(view, gaussians, pipeline, bg)
            obj_index_image_before = render_pkg["obj_index"]
            depth = render_pkg["surf_depth"][0]
            gt_obj_one_hot_image = view.original_obj_one_hot_image.cuda() 
            FoVx = view.FoVx
            focal = fov2focal(FoVx, view.image_width)
            rendering_before = render_pkg["render"]
            mask_before = render_pkg["seg"]
            torchvision.utils.save_image(rendering_before, os.path.join(render_path, image_name + ".png"))
            torchvision.utils.save_image(mask_before, os.path.join(gts_path, image_name + ".png"))

        start_align_time = clocktime()
        # C, H, W = obj_index_image_before.shape
        # point_idx = 0
        # xyz = gaussians._xyz.detach().cpu().numpy()
        # _obj_index = gaussians.get_obj_index

        # yy, xx = torch.meshgrid(torch.arange(H, device=obj_index_image_before.device),
        #                         torch.arange(W, device=obj_index_image_before.device),
        #                         indexing='ij') 
        # for c in (sort_index):
        #     mask = (obj_index_image_before[c] > 0.5)
        #     mask_= (gt_obj_one_hot_image[c] > 0.5)
        #     mask_f = mask.float()
        #     y_center = (yy * mask_f).sum() / mask_f.sum()
        #     x_center = (xx * mask_f).sum() / mask_f.sum()
        #     mask_f_ = mask_.float()
        #     y_center_ = (yy * mask_f_).sum() / mask_f_.sum()
        #     x_center_ = (xx * mask_f_).sum() / mask_f_.sum()
        #     target_depth = depth[int(y_center.item()),int(x_center.item())]
        #     mask = (_obj_index[:,c]>0.50) # *(gaussians._xyz[:,2]>0)
        #     mask = mask.detach().cpu().numpy()
        #     obj_pc_after[point_idx:point_idx+sum(mask)][:,0] = obj_pc_after[point_idx:point_idx+sum(mask)][:,0] -((y_center_-y_center)*target_depth/focal).item()
        #     obj_pc_after[point_idx:point_idx+sum(mask)][:,1] = obj_pc_after[point_idx:point_idx+sum(mask)][:,1] -((x_center_-x_center)*target_depth/focal).item()
        #     point_idx+=sum(mask)
        end_align_time = clocktime()
        print(f"Align Time : {round(end_align_time - start_align_time, 3)}")

        gaussians.save_ply_full(obj_pc_after, gaussians.visible_obj_index, os.path.join(args.model_path, f"point_cloud_{int(time+1):02d}.ply")) 
    return end_mpm_time - start_mpm_time, end_align_time - start_align_time


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--time", default=0, type=float)
    parser.add_argument("--sim_obj", "-o", nargs="+", type=int, default=[0])

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # physim(model.extract(args), args.iteration, pipeline.extract(args), args.obj_index, args.time, args.sim_obj)

    starttime = clocktime()
    mpm_time, align_time = physim_pp(model.extract(args), args.iteration, pipeline.extract(args), args.time)
    endtime = clocktime()
    print(f"Physics Simulation (mpm+align) Time : {round(mpm_time, 3)} + {round(align_time, 3)} / {round(endtime - starttime, 3)}")
    json_path = os.path.join(args.model_path, "perform_time.json")
    
    with open(json_path, 'r') as f:
        json_info = json.load(f)

    json_info[f"physim_{args.time}_full"] = round(endtime - starttime, 3)
    json_info[f"physim_{args.time}_mpm"] = round(mpm_time, 3)
    json_info[f"physim_{args.time}_align"] = round(align_time, 3)
    with open(json_path, 'w') as f:
        json.dump(json_info, f, indent=4)