import json
import os
import warnings
import cv2
import math
import numpy as np
import sys
import shutil
import argparse

from colors import colors

# only used for real dataset (hard-coded)
HAND_EYE_INIT = np.array([
    [-0.00655526, -0.999961  ,  0.00593217,  0.092329  ],
    [ 0.999813  , -0.006662  , -0.0181578 ,  0.00308126],
    [ 0.0181966 ,  0.00581203,  0.999818  ,  0.0622155 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
CAMERA_INTRINSICS = np.array([ 
    [882.17078849, 0.0, 639.5], 
    [0.0, 882.35647147, 359.5], 
    [0.0, 0.0, 1.0]
])

def mat2dict(mat, w, h):
    fl_x=mat[0,0]
    fl_y= mat[1,1]
    camera_angle_x = math.atan(w / (fl_x * 2)) * 2
    camera_angle_y = math.atan(h / (fl_y * 2)) * 2
    cx=mat[0,2]
    cy= mat[1,2]
    cam = {		
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
    }
    return cam

np.random.seed(seed=0)

parser = argparse.ArgumentParser(description='R6D')
parser.add_argument(
    '--input_dir', required=True,
    default="", type=str,
    help="input directory including syn_test_xx"
)
parser.add_argument(
    '--dataset', required=False,
    default="syn", type=str,
    help="dataset"
)
args = parser.parse_args()

for time in [0,1]:
    if args.dataset == "syn":
        data_dir = os.path.join(args.input_dir, f"train_{time}/data")
        frames_dir = os.path.join(data_dir, "rgb")
    elif args.dataset == "real":
        data_dir = os.path.join(args.input_dir, f"train_{time}/camera")
        frames_dir = os.path.join(data_dir, "RGB")
    else:
        raise ValueError("dataset should be 'syn' or 'real'")

    depth_dir = os.path.join(data_dir, "depth")
    mask_dir = os.path.join(data_dir, "color_mask")
    bmask_dir = os.path.join(data_dir, "binary_mask")
    class_dir = os.path.join(data_dir, "class")
    # bmask_each_dir = os.path.join(data_dir, "binary_mask_each")

    mask_image_files = [f for f in sorted(os.listdir(mask_dir)) if f.endswith(".png")]
    obj_idx_path = os.path.join(data_dir, "obj_idx.json") if time == 0 else os.path.join(data_dir, "obj_idx_t1.json")
    with open(obj_idx_path) as f:
        index_list = json.load(f)

    json_data = {		
        "num_obj": len(index_list),
        "frames": [],
    }

    if args.dataset == "syn":
        with open(os.path.join(data_dir, 'scene_camera.json')) as f:
            cam_poses = json.load(f)
        json_data["hand_eye_init"] = np.eye(4).tolist()
    elif args.dataset == "real":
        pose_dir = os.path.join(args.input_dir, f"train_{time}/pose")
        json_data["hand_eye_init"] = HAND_EYE_INIT.tolist()
    print(json_data)

    for idx, image_file in enumerate(mask_image_files):
        if args.dataset == "real" and idx % 4 != 0:
            continue

        image_name = image_file.split('.')[0] # idx:06d
        image_path = os.path.join(mask_dir, f"{idx:06d}.png")
        binary_mask_path = os.path.join(bmask_dir, f"{idx:06d}.png")
        class_path = os.path.join(class_dir, f"{idx:06d}.png")

        if args.dataset == "syn":
            original_image_path = os.path.join(frames_dir, f"{idx:06d}.png")
            depth_path = os.path.join(depth_dir, f"{idx:06d}.png")
            cam_K = np.array(cam_poses[str(int(image_name))]['cam_K']).reshape(3,3)
            cam_rot = np.array(cam_poses[str(int(image_name))]['cam_R_w2c']).reshape(3,3)
            cam_tra = np.array(cam_poses[str(int(image_name))]['cam_t_w2c'])*0.001
            matrix_values = np.concatenate((cam_rot,cam_tra.reshape(3,1)), axis = 1)
            matrix_values = np.vstack((matrix_values, np.array([0, 0, 0, 1])))
            matrix_values = np.array(matrix_values)
            pose = np.linalg.inv(matrix_values)
        elif args.dataset == "real":
            original_image_path = os.path.join(frames_dir, f"{(idx+1):05d}.png")
            depth_path = os.path.join(depth_dir, f"{(idx+1):05d}.png")
            cam_K = CAMERA_INTRINSICS
            pose_path = os.path.join(pose_dir, f"{(idx+1):05d}.txt")
            with open(pose_path) as f:
                cam_pose = f.read()
            pose = np.array([float(i) for i in cam_pose.split()]).reshape(4,4)

        if idx == 0:
            img = cv2.imread(original_image_path)
            H, W, C = img.shape    
        
        cam = mat2dict(cam_K, W, H)
        new_frame = {
            "file_path": image_path,
            "original_path": original_image_path,
            "binary_mask_path": binary_mask_path,
            "class_path": class_path,
            "transform_matrix": pose.tolist(),
            "depth_path": depth_path,
            "cam": cam,
            "obj_idx": index_list,
        }
        json_data["frames"].append(new_frame)

    if time == 0:
        with open(os.path.join(args.input_dir, "transforms_sparse_train.json"), 'w') as f:
            json.dump(json_data, f, indent="\t")
    else: # time == 1
        with open(os.path.join(args.input_dir, "transforms_sparse_train_01.json"), 'w') as f:
            json.dump(json_data, f, indent="\t")

for time in [0,1]:
    if args.dataset == "syn":
        data_dir = os.path.join(args.input_dir, f"test_{time}/data")
        obj_idx_dir = os.path.join(args.input_dir, f"train_{time}/data")
        frames_dir = os.path.join(data_dir, "rgb")
    elif args.dataset == "real":
        data_dir = os.path.join(args.input_dir, f"train_{time}/camera")
        obj_idx_dir = os.path.join(args.input_dir, f"train_{time}/camera")
        frames_dir = os.path.join(data_dir, "RGB")
    else:
        raise ValueError("dataset should be 'syn' or 'real'")

    depth_dir = os.path.join(data_dir, "depth")
    mask_dir = os.path.join(data_dir, "color_mask")
    bmask_dir = os.path.join(data_dir, "binary_mask")
    class_dir = os.path.join(data_dir, "class")
    # bmask_each_dir = os.path.join(data_dir, "binary_mask_each")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(bmask_dir, exist_ok=True)
    os.makedirs(class_dir, exist_ok=True)
    # os.makedirs(bmask_each_dir, exist_ok=True)

    mask_image_files = [f for f in sorted(os.listdir(mask_dir)) if f.endswith(".png")]
    obj_idx_path = os.path.join(obj_idx_dir, "obj_idx.json") if time == 0 else os.path.join(obj_idx_dir, "obj_idx_t1.json")
    with open(obj_idx_path) as f:
        index_list = json.load(f)

    json_data = {		
        "num_obj": len(index_list),
        "frames": [],
    }

    if args.dataset == "syn":
        with open(os.path.join(data_dir, 'scene_camera.json')) as f:
            cam_poses = json.load(f)
        json_data["hand_eye_init"] = np.eye(4).tolist()
    elif args.dataset == "real":
        pose_dir = os.path.join(args.input_dir, f"train_{time}/pose")
        json_data["hand_eye_init"] = HAND_EYE_INIT.tolist()

    image_files = [f for f in sorted(os.listdir(frames_dir)) if f.endswith(".png")]

    for idx, image_file in enumerate(image_files):
        if args.dataset == "real" and idx % 4 == 0:
            continue

        image_name = image_file.split('.')[0] # idx:06d

        if args.dataset == "syn":
            original_image_path = os.path.join(frames_dir, f"{idx:06d}.png")
            depth_path = os.path.join(depth_dir, f"{idx:06d}.png")
            cam_K = np.array(cam_poses[str(int(image_name))]['cam_K']).reshape(3,3)
            cam_rot = np.array(cam_poses[str(int(image_name))]['cam_R_w2c']).reshape(3,3)
            cam_tra = np.array(cam_poses[str(int(image_name))]['cam_t_w2c'])*0.001
            matrix_values = np.concatenate((cam_rot,cam_tra.reshape(3,1)), axis = 1)
            matrix_values = np.vstack((matrix_values, np.array([0, 0, 0, 1])))
            matrix_values = np.array(matrix_values)
            pose = np.linalg.inv(matrix_values)

            mask_image_path = os.path.join(mask_dir, f"{idx:06d}.png")
            binary_mask_path = os.path.join(bmask_dir, f"{idx:06d}.png")
            class_path = os.path.join(class_dir, f"{idx:06d}.png")

            img = cv2.imread(original_image_path,-1)
            H,W,C = img.shape
            color_mask = np.zeros((H, W, 3), dtype=np.uint8)
            binary_mask = np.zeros((H, W), dtype=np.uint8)
            class_image = np.zeros((H, W), dtype=np.uint8)
            for k in range(len(index_list)):
                try:
                    mask_image = cv2.imread(os.path.join(data_dir, 'mask_visib', image_name + f"_{k:06d}"+".png"),-1)
                    mask = mask_image > 0
                    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
                    color_mask[:, :, 0] = colors[int(k)-1, 0]*255
                    color_mask[:, :, 1] = colors[int(k)-1, 1]*255
                    color_mask[:, :, 2] = colors[int(k)-1, 2]*255
                    img[mask!=0] = img[mask!=0].astype(np.float32) * 0.3 + color_mask[mask!=0].astype(np.float32) * 0.7
                    binary_mask[mask!=0] = 255
                    class_image[mask!=0] = (k+1)
                except:
                    continue
            cv2.imwrite(mask_image_path, img)
            cv2.imwrite(binary_mask_path, binary_mask)
            cv2.imwrite(class_path, class_image)
                
            cam = mat2dict(cam_K, W, H)
            new_frame = {
                "file_path": mask_image_path,
                "original_path": original_image_path,
                "binary_mask_path": binary_mask_path,
                "class_path": class_path,
                "transform_matrix": pose.tolist(),
                "depth_path": depth_path,
                "cam": cam,
                "obj_idx": index_list,
            }
            
        elif args.dataset == "real":
            original_image_path = os.path.join(frames_dir, f"{(idx+1):05d}.png")
            depth_path = os.path.join(depth_dir, f"{(idx+1):05d}.png")
            cam_K = CAMERA_INTRINSICS
            pose_path = os.path.join(pose_dir, f"{(idx+1):05d}.txt")
            with open(pose_path) as f:
                cam_pose = f.read()
            pose = np.array([float(i) for i in cam_pose.split()]).reshape(4,4)
            cam = mat2dict(cam_K, W, H)
            new_frame = {
                "original_path": original_image_path,
                "transform_matrix": pose.tolist(),
                "cam": cam,
                "obj_idx": index_list,
            }
        
        json_data["frames"].append(new_frame)

    if time == 0:
        with open(os.path.join(args.input_dir, "transforms_sparse_test.json"), 'w') as f:
            json.dump(json_data, f, indent="\t")
    else: # time == 1
        with open(os.path.join(args.input_dir, "transforms_sparse_test_01.json"), 'w') as f:
            json.dump(json_data, f, indent="\t")