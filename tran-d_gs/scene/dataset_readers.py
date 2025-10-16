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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2
np.random.seed(42)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array
    seg: np.array
    obj_image: np.array
    visible_obj_index: np.array
    total_obj_num: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

# not used
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)  
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    shs_seg = np.vstack([vertices['seg1'], vertices['seg2'], vertices['seg3']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    i = 0
    while f"obj_index{i}" in vertices:
        obj_name = f"obj_index{i}"
        if i == 0:
            obj_index = vertices[obj_name]
        else:
            obj_index = np.vstack([obj_index, vertices[obj_name]])
        i += 1
    obj_index = obj_index.T
    return BasicPointCloud(points=positions, colors=colors, seg= shs_seg, normals=normals, obj_index = obj_index)

def storePly(path, xyz, rgb, seg, obj_index, num_obj):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('seg1', 'u1'), ('seg2', 'u1'), ('seg3', 'u1'),]
    
    for i in range(num_obj):
        dtype+=[(f'obj_index{i}', 'f4')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, seg,obj_index), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# not used
def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)


    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", time=0, views=6):
    cam_infos = []
    if time == 0.5:
        if "train" in transformsfile:
            transformsfile = "transforms_sparse_train_01.json"
        if "test" in transformsfile:
            transformsfile = "transforms_sparse_test_01.json"
    elif time != 0: # time = 1
        if "train" in transformsfile:
            transformsfile = f"transforms_sparse_train_{time:02d}.json"
        if "test" in transformsfile:
            transformsfile = f"transforms_sparse_test_{time:02d}.json"

    print(os.path.join(path, transformsfile))
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

    frames = contents["frames"]
    hand_eye = np.array(contents["hand_eye_init"])
    total_obj_num = contents["num_obj"]

    bg_val = np.array([255, 255, 255] if white_background else [0, 0, 0], dtype=np.uint8)

    for idx, frame in enumerate(frames):
        # syn dataset's train json has 13 views (1 BEV + 12 spiral). We only use spiral views for training (skip 0th view)
        if "syn" in path:
            if "train" in transformsfile and time == 0 :
                if views == 12:
                    if idx == 0:
                        continue
                else:
                    if idx % (12 / views) != 1:
                        continue

        # for updating in t = 1, we only use the first frame for continual training
        if "train" in transformsfile and time != 0 :
            if idx != 0:
                continue
                
        fovx = frame["cam"]["camera_angle_x"]
        cam_name = frame["original_path"]
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem

        im_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if im_data.shape[2] == 3:
            im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB)
            alpha = np.full(im_data.shape[:2], 255, dtype=np.uint8)
            im_data = np.dstack((im_data, alpha))
        else:
            im_data[:, :, :3] = cv2.cvtColor(im_data[:, :, :3], cv2.COLOR_BGR2RGB)
        im_data = im_data.astype(np.float32)

        if "depth_path" in frame:
            depth = cv2.imread(frame["depth_path"].replace('.JPG', '.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        else:
            depth = np.ones(im_data.shape[:2], dtype=np.float32)

        if "file_path" in frame:
            mask_data = cv2.imread(os.path.join(path, frame["file_path"]), cv2.IMREAD_UNCHANGED)
            if mask_data.shape[2] == 3:
                mask_data = cv2.cvtColor(mask_data, cv2.COLOR_BGR2RGB)
                alpha = np.full(mask_data.shape[:2], 255, dtype=np.uint8)
                mask_data = np.dstack((mask_data, alpha))
            else:
                mask_data[:, :, :3] = cv2.cvtColor(mask_data[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            mask_data = im_data.copy()
        
            
        if "class_path" in frame:
            obj_image = cv2.imread(os.path.join(path, frame["class_path"]), cv2.IMREAD_UNCHANGED)
            one_hot_image = np.eye(15, dtype=np.float32)[obj_image]
            bg_mask = one_hot_image[:, :, 0] == 1
            im_data[bg_mask] = np.append(bg_val, 255)
            mask_data[bg_mask] = np.append(bg_val, 255)
        else:
            obj_image = np.ones_like(cv2.imread(image_path,0))
            one_hot_image = np.eye(15, dtype=np.float32)[obj_image]

        visible_obj_index = np.array(frame["obj_idx"])

        c2w = np.array(frame["transform_matrix"])@hand_eye
        w2c = np.linalg.inv(c2w)
        R = w2c[:3,:3].T
        T = w2c[:3, 3]

        image = Image.fromarray(im_data[:, :, :3].astype(np.uint8), "RGB")
        mask = Image.fromarray(mask_data[:, :, :3].astype(np.uint8), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        cam_infos.append(CameraInfo(
            uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, 
            image=image, image_path=image_path, image_name=image_name, 
            width=image.size[0], height=image.size[1], 
            depth = depth, seg = mask, 
            obj_image = one_hot_image, visible_obj_index = visible_obj_index, total_obj_num = total_obj_num 
        ))

        # when making mesh for physic simulation at t=0.5, we add one view looking at the object from below
        # transform / hand-eye is hard-coded and other parameters are copied from the last frame (not used for rendering)
        if time == 0.5 and idx ==len(frames)-1:
            if "real" in path:
                transform = np.array([
                    [ 0.999860894854304, -0.0004257526395561435, -0.016673622157181524,0.3078897979053868],
                    [-0.00042519762366097637, -0.9999999089251849, 3.6832086719007845e-05, -4.517987763705683e-05],
                    [-0.0166736363199926, -2.9737378667163875e-05, -0.9998609848211685, 0.5842006150114663],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                hand_eye_init = np.array([
                    [-0.00655526, -0.999961, 0.00593217, 0.092329],
                    [0.999813, -0.006662, -0.0181578, 0.00308126],
                    [0.0181966, 0.00581203, 0.999818, 0.0622155],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                reflect = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32) @ transform @ hand_eye_init

            if "syn" in path:
                c2w = np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 1.0000001220703125],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                reflect = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32) @ c2w

            w2c = np.linalg.inv(reflect)
            R_ = np.transpose(w2c[:3,:3])
            T_ = w2c[:3, 3]  
            cam_infos.append(CameraInfo(
                uid=idx+1, R=R_, T=T_, FovY=fovy, FovX=fovx, 
                image=image, image_path=image_path, image_name=image_name, 
                width=image.size[0], height=image.size[1], 
                depth = depth, seg = mask, 
                obj_image = one_hot_image, visible_obj_index = visible_obj_index, total_obj_num =total_obj_num
            ))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, time, views,extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_sparse_train.json", white_background, extension, time, views)
    test_cam_infos = readCamerasFromTransforms(path, "transforms_sparse_test.json", white_background, extension, time, views)
    nerf_normalization = getNerfppNorm(train_cam_infos) 
    ply_path = os.path.join(path, "points3d_2d.ply")
    
    if not os.path.exists(ply_path):
        print("create random pcd")
        num_pts = 100_000
        bottom = np.random.uniform(low=[-0.5, -0.5, 0], high=[0.5, -0.5, 0.1], size=(num_pts//5, 3))
        front = np.random.uniform(low=[-0.5, -0.5, 0], high=[0.5, 0.5, 0.1], size=(num_pts // 5, 3))
        back = np.random.uniform(low=[-0.5, 0.5, 0], high=[0.5, 0.5, 0.1], size=(num_pts // 5, 3))
        left = np.random.uniform(low=[-0.5, -0.5, 0], high=[-0.5, 0.5, 0.1], size=(num_pts // 5, 3))
        right = np.random.uniform(low=[0.5, -0.5, 0], high=[0.5, 0.5, 0.1], size=(num_pts // 5, 3))
        xyz = np.concatenate((bottom, front, back, left, right), axis=0)

        shs = np.random.random((num_pts, 3)) / 255.0
        shs_seg = np.random.random((num_pts, 3)) / 255.0
        obj_index = np.random.random((num_pts, 15))/255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), seg= SH2RGB(shs_seg), normals=np.zeros((num_pts, 3)), obj_index = obj_index)
        storePly(ply_path, xyz, SH2RGB(shs) * 255, SH2RGB(shs_seg)*255, obj_index, 15)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
