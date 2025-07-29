import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import argparse
import string
import random
import json
import math
import time as tm
random.seed(23)
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    default=None,
) 
parser.add_argument('--index', required=True,
                    default=0, type=int,
                    help="index")  

opt = parser.parse_args()
colors = np.array([
[0.89, 0.28, 0.13],
[0.45, 0.38, 0.92],
[0.35, 0.73, 0.63],
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

images_dir = opt.input_dir
data_dir = images_dir.split("rgb")[0]
frames_dir = data_dir + 'RGB'
save_bbox = data_dir + 'bbox'
save_folder_mask = data_dir + 'mask_'
save_binary_mask_folder = data_dir + 'binary_mask'
save_class_folder = data_dir + 'class'
save_binary_each_mask_folder = data_dir + 'binary_mask_each'

GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "checkpoint_1.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.2
TEXT_PROMPT = "786dvpteg. object."
SOURCE_VIDEO_FRAME_DIR = frames_dir
SAVE_TRACKING_RESULTS_DIR = save_folder_mask
PROMPT_TYPE_FOR_VIDEO = "mask" # choose from ["point", "box", "mask"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)
frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".JPG", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)
ann_frame_idx = 0
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
image_source, image = load_image(img_path)
boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    remove_combined = True
)

import gc
del grounding_model
torch.cuda.empty_cache()
torch.cuda.synchronize()
gc.collect()


h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
confidences = confidences.numpy().tolist()
class_names = labels
image_predictor.set_image(image_source)
OBJECTS = class_names

# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

if masks.ndim == 4:
    masks = masks.squeeze(1)

img = cv2.imread(img_path)
class_ids = np.array(list(range(len(class_names))))
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

def process_masks_by_size(masks, threshold=1000):
    processed_masks = masks.copy()
    mask_sizes = [np.sum(mask) for mask in masks]
    sorted_indices = np.argsort(mask_sizes)
    sorted_masks = processed_masks[sorted_indices]
    for i in range(len(sorted_masks)):
        if np.sum(sorted_masks[i]) < threshold:
            continue
        for j in range(i + 1, len(sorted_masks)):
            overlap_region = np.logical_and(sorted_masks[j], sorted_masks[i])
            sorted_masks[j][overlap_region] = 0
    filtered_masks = [
        mask for mask in sorted_masks if np.sum(mask) >= threshold
    ]
    if len(filtered_masks) > 0:
        return np.array(filtered_masks)
    else:
        return np.empty((0, *masks.shape[1:]), dtype=masks.dtype)

masks = process_masks_by_size(masks)


assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

if PROMPT_TYPE_FOR_VIDEO == "point":
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
        labels = np.ones((points.shape[0]), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        labels = np.ones((1), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask
        )
else:
    raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

os.makedirs(SAVE_TRACKING_RESULTS_DIR, exist_ok=True)
os.makedirs(save_binary_mask_folder, exist_ok=True)
os.makedirs(save_class_folder, exist_ok=True)
os.makedirs(save_binary_each_mask_folder, exist_ok=True)
os.makedirs(save_bbox, exist_ok=True)
os.makedirs(save_folder_mask, exist_ok=True)


ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
image_paths = []
reindex = 0
index_list = []
num_obj = len(masks)
for frame_idx, segments in video_segments.items():
    save_path = os.path.join(save_folder_mask, f'{reindex:06d}.png')
    save_path_binary = os.path.join(save_binary_mask_folder, f'{reindex:06d}.png')
    save_path_class = os.path.join(save_class_folder, f'{reindex:06d}.png')
    img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))
    H,W,C = img.shape
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    class_image = np.zeros((H, W), dtype=np.uint8)
    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)
    class_id=np.array(object_ids, dtype=np.int32)
    for i in range(len(masks)):
        save_path_binary_each = os.path.join(save_binary_each_mask_folder, f'{reindex:06d}_{i:06d}.png')
        binary_mask_each = np.zeros((H, W), dtype=np.uint8)
        mask_ = np.stack((masks[i]>0, masks[i]>0, masks[i]>0), axis = -1)
        binary_mask[masks[i]!=0] = 255
        binary_mask_each[masks[i]!=0] = 255
        class_image[masks[i]!=0] = (int(class_id[i]))
        cv2.imwrite(save_path_binary_each, binary_mask_each)
        color = colors[class_id[i]]
        mask_image = masks[i].reshape(h, w, 1)* color.reshape(1, 1, -1)
        img = img*(1-mask_) + mask_image[:,:,:3]*mask_*255.0*0.7 + img[:,:,:3]*mask_*0.3
        if frame_idx==0:
            index_list.append(i+1)
    cv2.imwrite(save_path_binary, binary_mask)
    cv2.imwrite(save_path_class, class_image)
    cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"{reindex:06d}.png"), img)
    reindex+=1

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
            "num_obj": num_obj,
			"frames": [],
		}

out_ = {		
            "num_obj": num_obj,
			"frames": [],
		}

seq_index = f"syn_test_{opt.index:02d}"
dynamic_cam_image_folder = os.path.join(f"data/{seq_index}","train_0/rgb")
dynamic_cam_mask_folder = os.path.join(f"data/{seq_index}","train_0/mask_")
dynamic_cam_binary_mask_folder = os.path.join(f"data/{seq_index}","train_0/binary_mask")
dynamic_cam_class_folder = os.path.join(f"data/{seq_index}","train_0/class")
dynamic_cam_depth = os.path.join(f"data/{seq_index}","train_0/depth")
pose_folder = os.path.join(f"data/{seq_index}","pose")

dynamic_image_files = [f for f in sorted(os.listdir(dynamic_cam_mask_folder)) if f.endswith(".png")]

with open(os.path.join(f"data/{seq_index}",f"train_0/data",'scene_camera.json')) as f:
    cam_poses = json.load(f)


for idx, image_file in enumerate(dynamic_image_files):
    image_name = image_file.split('.')[0]
    image_path = os.path.join(dynamic_cam_mask_folder,image_name+".png") 
    original_image_path = os.path.join(dynamic_cam_image_folder,f"{idx:06d}"+".png") 
    binay_mask_path = os.path.join(dynamic_cam_binary_mask_folder,image_name+".png") 
    class_path = os.path.join(dynamic_cam_class_folder,image_name+".png")
    cam_K = np.array(cam_poses[str(int(image_name))]['cam_K']).reshape(3,3)
    cam_rot = np.array(cam_poses[str(int(image_name))]['cam_R_w2c']).reshape(3,3)
    cam_tra = np.array(cam_poses[str(int(image_name))]['cam_t_w2c'])*0.001
    matrix_values = np.concatenate((cam_rot,cam_tra.reshape(3,1)), axis = 1)
    matrix_values = np.vstack((matrix_values, np.array([0, 0, 0, 1])))
    matrix_values = np.array(matrix_values)
    pose_openCV = np.linalg.inv(matrix_values)
    pose = pose_openCV
    pose_list = pose.tolist()
    depth = os.path.join(dynamic_cam_depth,f"{idx:06d}"+".png")
    cam = mat2dict(cam_K, W, H)
    new_frame = {
    "file_path": f"{image_path}",
    "original_path": f"{original_image_path}",
    "binary_mask_path": f"{binay_mask_path}",
    "class_path": f"{class_path}",
    "transform_matrix": pose_list,
    "depth_path": f"{depth}",
    "cam": cam,
    "obj_idx": index_list,
    }
    out["frames"].append(new_frame)

with open(os.path.join(f"data/{seq_index}","transforms_sparse_train.json"), 'w') as f:
   json.dump(out, f, indent="\t")


