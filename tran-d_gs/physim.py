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
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from time import time as clocktime
import json
import open3d as o3d

def physim_pp(dataset : ModelParams, iteration : int, pipeline : PipelineParams, time : int):
    obj_idxs = []
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, time, load_iteration=iteration, shuffle=False)
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(0)
    gaussians.visible_obj_index = viewpoint_cam.visible_obj_index
    real_gaussian_particle_size_list = []
    bg_color = [1,1,1]

    gaussExtractor = GaussianExtractor(gaussians, render, pipeline, bg_color=bg_color)
    gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(scene.getTestCameras())
    depth_trunc = (gaussExtractor.radius * 2.0)
    voxel_size = (depth_trunc / 256)
    sdf_trunc = 5.0 * voxel_size
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    mesh_post = post_process_mesh(mesh, cluster_to_keep=50)
    mesh_path = os.path.join(scene.model_path, "test", "ours_{}_before_mpm".format(iteration))
    os.makedirs(mesh_path, exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(mesh_path, "mesh.ply"), mesh_post)
    
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
            mask = np.all(colors == np.array([i/255, i/255, i/255]), axis=1)
            points_of_idx = points[mask]

            real_gaussian_particle = gaussians.get_xyz_each_obj_index(i).detach().cpu().numpy()
            real_gaussian_particle_size = real_gaussian_particle.shape[0]
            real_gaussian_particle_size_list.append(real_gaussian_particle_size)
            rigid_idx = np.zeros(real_gaussian_particle_size, dtype=bool)
            all_particles = np.concatenate([real_gaussian_particle, points_of_idx], axis=0)

            particles = all_particles.astype('float32')
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
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        bg = background
        render_path = os.path.join(scene.model_path, "test", "ours_{}_before_mpm".format(iteration), "renders")
        gts_path = os.path.join(scene.model_path, "test", "ours_{}_before_mpm".format(iteration), "masked")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)

        for idx, view in enumerate(tqdm(scene.getTrainCameras(), desc="Rendering progress")):
            image_name = view.image_name             
            render_pkg = render(view, gaussians, pipeline, bg)
            rendering_before = render_pkg["render"]
            mask_before = render_pkg["seg"]
            torchvision.utils.save_image(rendering_before, os.path.join(render_path, image_name + ".png"))
            torchvision.utils.save_image(mask_before, os.path.join(gts_path, image_name + ".png"))

        gaussians.save_ply_full(obj_pc_after, gaussians.visible_obj_index, os.path.join(args.model_path, f"point_cloud_{int(time+1):02d}.ply")) 
    return end_mpm_time - start_mpm_time

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

    starttime = clocktime()
    mpm_time = physim_pp(model.extract(args), args.iteration, pipeline.extract(args), args.time)
    endtime = clocktime()
    print(f"Physics Simulation mpm / full Time : {round(mpm_time, 3)} / {round(endtime - starttime, 3)}")
    json_path = os.path.join(args.model_path, "perform_time.json")
    
    with open(json_path, 'r') as f:
        json_info = json.load(f)

    json_info[f"physim_{args.time}_full"] = round(endtime - starttime, 3)
    json_info[f"physim_{args.time}_mpm"] = round(mpm_time, 3)
    with open(json_path, 'w') as f:
        json.dump(json_info, f, indent=4)