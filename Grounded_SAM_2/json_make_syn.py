import json
import os
import warnings
import cv2
import math
import numpy as np
import sys
import shutil
np.random.seed(seed=0)
import argparse
parser = argparse.ArgumentParser(description='R6D')
parser.add_argument('-i', '--index', required=True,
                    default=0, type=int,
                    help="index")  
parser.add_argument('-o', '--object_num', required=True,
                    default=0, type=int)  

parser.add_argument('-t', '--time', required=True,
                    default=0, type=int)  

args = parser.parse_args()
colors = np.array([
  [0.89, 0.28, 0.13],
  [0.45, 0.38, 0.92],
  [0.35, 0.73, 0.63],
  [0.62, 0.28, 0.91],
  [0.65, 0.71, 0.22],
  [0.8, 0.29, 0.89],
  [0.27, 0.55, 0.22],
  [0.37, 0.46, 0.84],
  [0.84, 0.63, 0.22],
  [0.68, 0.29, 0.71],
  [0.48, 0.75, 0.48],
  [0.88, 0.27, 0.75],
  [0.82, 0.45, 0.2],
  [0.86, 0.27, 0.27],
  [0.52, 0.49, 0.18],
  [0.33, 0.67, 0.25],
  [0.67, 0.42, 0.29],
  [0.67, 0.46, 0.86],
  [0.36, 0.72, 0.84],
  [0.85, 0.29, 0.4],
  [0.24, 0.53, 0.55],
  [0.85, 0.55, 0.8],
  [0.4, 0.51, 0.33],
  [0.56, 0.38, 0.63],
  [0.78, 0.66, 0.46],
  [0.33, 0.5, 0.72],
  [0.83, 0.31, 0.56],
  [0.56, 0.61, 0.85],
  [0.89, 0.58, 0.57],
  [0.67, 0.4, 0.49]
])   

def mat2dict(mat, w, h):
    fl_x=mat[0,0]
    fl_y= mat[1,1]
    camera_angle_x = math.atan(w / (fl_x * 2)) * 2
    camera_angle_y = math.atan(h / (fl_y * 2)) * 2
    cx=mat[0,2]
    cy= mat[1,2]
    cam = {		"camera_angle_x": camera_angle_x,
                "camera_angle_y": camera_angle_y,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
            }
    return cam

out = {		
            "num_obj": args.object_num,
			"frames": []
		}
out_ = {
            "num_obj": args.object_num,
			"frames": []
		}

seq_index = f"syn_multi_test_{args.index:02d}"
test_path =  f"/mydata/jyk/ICCV2025/{seq_index}"
syn_folder = os.path.join(test_path,f"test_{args.time:d}/data")
dynamic_cam_image_folder = os.path.join(test_path,f"test_{args.time:d}/data/rgb")
dynamic_cam_mask_folder = os.path.join(test_path,f"test_{args.time:d}/data/mask_")
dynamic_cam_binary_mask_folder = os.path.join(test_path,f"test_{args.time:d}/data/binary_mask")
dynamic_cam_class_folder = os.path.join(test_path,f"test_{args.time:d}/data/class")
dynamic_cam_depth = os.path.join(test_path,f"test_{args.time:d}/data/depth")

os.makedirs(dynamic_cam_mask_folder, exist_ok=True)
os.makedirs(dynamic_cam_binary_mask_folder, exist_ok=True)
os.makedirs(dynamic_cam_class_folder, exist_ok=True)
index_list = []
image_files = [f for f in sorted(os.listdir(dynamic_cam_image_folder)) if f.endswith(".png")]
with open(os.path.join(os.path.join(test_path,f"test_{args.time:d}/data"),'scene_camera.json')) as f:
    cam_poses = json.load(f)

for idx, image_file in enumerate(image_files):
    image_name = image_file.split('.')[0]
    cam_K = np.array(cam_poses[str(int(image_name))]['cam_K']).reshape(3,3)
    cam_rot = np.array(cam_poses[str(int(image_name))]['cam_R_w2c']).reshape(3,3)
    cam_tra = np.array(cam_poses[str(int(image_name))]['cam_t_w2c'])*0.001
    matrix_values = np.concatenate((cam_rot,cam_tra.reshape(3,1)), axis = 1)
    matrix_values = np.vstack((matrix_values, np.array([0, 0, 0, 1])))
    matrix_values = np.array(matrix_values)
    pose_openCV = np.linalg.inv(matrix_values)
    pose = pose_openCV
    pose_list = pose.tolist()
    train_mask_path = os.path.join(dynamic_cam_mask_folder,image_name+".png")
    train_binary_mask_path = os.path.join(dynamic_cam_binary_mask_folder,image_name+".png")
    class_path = os.path.join(dynamic_cam_class_folder,image_name+".png")
    cam_depth_path = os.path.join(dynamic_cam_depth,image_name+".png")
    origin_image_path = os.path.join(dynamic_cam_image_folder,image_name+".png")

    img = cv2.imread(origin_image_path,-1)
    H,W,C = img.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    class_image = np.zeros((H, W), dtype=np.uint8)
    for k in range(args.object_num):
        try:
            mask_image = cv2.imread(os.path.join(syn_folder,'mask_visib', image_name + f"_{k:06d}"+".png"),-1)
            mask = mask_image>0
            color_mask = np.zeros((H, W, 3), dtype=np.uint8)
            color_mask[:, :, 0] = colors[int(k)-1, 0]*255
            color_mask[:, :, 1] = colors[int(k)-1, 1]*255
            color_mask[:, :, 2] = colors[int(k)-1, 2]*255
            img[mask!=0] = img[mask!=0].astype(np.float32) * 0.3 + color_mask[mask!=0].astype(np.float32) * 0.7
            binary_mask[mask!=0] = 255
            class_image[mask!=0] = (k+1)
        except:
            continue
    cv2.imwrite(train_mask_path, img)
    cv2.imwrite(train_binary_mask_path, binary_mask)
    cv2.imwrite(class_path, class_image)

        
    cam = mat2dict(cam_K, W, H)
    new_frame = {
    "file_path": f"{train_mask_path}",
    "depth_path": f"{cam_depth_path}",
    "original_path": f"{origin_image_path}",
    "class_path": f"{class_path}",
    "binary_mask_path": f"{train_binary_mask_path}",
    "transform_matrix": pose_list,
    "cam": cam,
    "obj_idx": index_list,
    }
    out["frames"].append(new_frame)
syn_folder = os.path.join(test_path,f"train_{args.time:d}/data")

if args.time!=0:
    with open(os.path.join(test_path,f"transforms_sparse_test_{args.time:02d}.json"), 'w') as f:
        json.dump(out, f, indent="\t")
else:
    with open(os.path.join(test_path,"transforms_sparse_test.json"), 'w') as f:
        json.dump(out, f, indent="\t")




