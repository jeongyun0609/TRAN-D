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

K = np.array([ [906.4620361328125, 0.0, 645.7659912109375], 
                [0.0, 906.65283203125,  375.2723388671875], 
                [0.0,       0.0,        1.0]])

def mat2dict(mat):
    fl_x=mat[0,0]
    fl_y= mat[1,1]
    w= 1280.0
    h= 720.0
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

hand_eye_init = np.array([[-0.00655526, -0.999961  ,  0.00593217,  0.092329  ],
                            [ 0.999813  , -0.006662  , -0.0181578 ,  0.00308126],
                            [ 0.0181966 ,  0.00581203,  0.999818  ,  0.0622155 ],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]])
out = {		
            "num_obj": args.object_num,
            "hand_eye_init": hand_eye_init.tolist(),
			"frames": []
		}
out_ = {
            "num_obj": args.object_num,
            "hand_eye_init": hand_eye_init.tolist(),
			"frames": []
		}

seq_index = f"real_multi_test_{args.index:02d}"
test_path =  f"/mydata/jyk/ICCV2025/{seq_index}"
syn_folder = os.path.join(test_path,f"train_{args.time:d}/camera")
dynamic_cam_image_folder = os.path.join(test_path,f"train_{args.time:d}/camera/rgb")
pose_folder = os.path.join(test_path,f"train_{args.time:d}/pose")
image_files = [f for f in sorted(os.listdir(dynamic_cam_image_folder)) if f.endswith(".JPG")]


for idx, image_file in enumerate(image_files):
    image_name = image_file.split('.')[0]
    pose_idx = idx+1
    pose_txt = os.path.join(pose_folder, f"{pose_idx:05d}"+".txt")
    with open(pose_txt) as f:
        cam_pose = f.read()
    matrix_values = np.array([float(i) for i in cam_pose.split()]).reshape(4,4)
    pose_list = matrix_values.tolist()
    cam = mat2dict(K)
    origin_image_path = os.path.join(dynamic_cam_image_folder,image_name+".JPG")
    index_list = []
    new_frame = {
    "original_path": f"{origin_image_path}",
    "transform_matrix": pose_list,
    "cam": cam,
    "obj_idx": index_list,
    }
    if idx%4==0:
        out_["frames"].append(new_frame)
    else:
        out["frames"].append(new_frame)

if args.time!=0:
    with open(os.path.join(test_path,f"transforms_sparse_test_{args.time:02d}.json"), 'w') as f:
        json.dump(out, f, indent="\t")
else:
    with open(os.path.join(test_path,"transforms_sparse_test.json"), 'w') as f:
        json.dump(out, f, indent="\t")




