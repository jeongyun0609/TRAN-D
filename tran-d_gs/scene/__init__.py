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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, time = 0, load_iteration=None, shuffle=True, resolution_scales=[1.0], only_objects=False, views = 6):
        self.model_path = args.model_path
        self.source_path = args.source_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if time != 0 and load_iteration is None:
            print("WARNING: time is set but load_iteration is not set, setting to max iteration")
            load_iteration = -1

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
            if only_objects:
                print("Loading only objects")

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            assert False, "COLMAP data is not supported. Make json file and use Blender format."
            # scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_sparse_train.json")):    
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, time, views)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print('Loaded Training Cameras ( res ', resolution_scale, '): ', len(self.train_cameras[resolution_scale]))
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print('Loaded Test Cameras ( res ', resolution_scale, '): ', len(self.test_cameras[resolution_scale]))

        if self.loaded_iter and shuffle:
            print("Load ", os.path.join(self.model_path, f"point_cloud_{time:02d}.ply"))
            self.gaussians.load_ply_training(os.path.join(self.model_path, f"point_cloud_{time:02d}.ply"), scene_info.train_cameras)
        elif self.loaded_iter and not shuffle and type(time) == int and time != 0:
            print("Load ", os.path.join(self.model_path, f"point_cloud_{time:02d}.ply"))
            self.gaussians.load_ply_training(os.path.join(self.model_path, f"point_cloud_{time:02d}.ply"), scene_info.train_cameras)
        elif self.loaded_iter:
            point_cloud_name = "point_cloud.ply" if not only_objects else "point_cloud_obj.ply"
            print("Load ", os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), point_cloud_name))
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), point_cloud_name))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        if args.remove_obj:
            visible_index = scene_info.train_cameras[0].visible_obj_index
            total_obj_num = scene_info.train_cameras[0].total_obj_num
            all_indices = set(range(1, total_obj_num + 1))
            missing_indices = list(all_indices - set(visible_index))
            print("Missing indices: ", missing_indices)
            visible_index = scene_info.train_cameras[0].visible_obj_index
            self.gaussians.load_ply_remove_obj(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"), missing_indices)

    def save(self, iteration, save_obj = False):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if save_obj:
            self.gaussians.save_ply_except_ground(os.path.join(point_cloud_path, "point_cloud_obj.ply"))

    def save_groups(self, iteration, groups, objs):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        for obj in objs:
            self.gaussians.save_groups_ply(os.path.join(point_cloud_path, f"vis_point_cloud_{obj}.ply"), groups, [obj])

    def save_objs(self, iteration, objs):
        for obj in objs:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_obj_ply(os.path.join(point_cloud_path, f"point_cloud_{obj}.ply"), obj)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]