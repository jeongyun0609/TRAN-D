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
import jsonlines
import math
import time as tm

from colors import colors
random.seed(23)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir",
    type=str,
    default=None,
) 
parser.add_argument(
    '--gd_checkpoint', required=False,
    default="./checkpoints/tran-d_seg_ckpt.pth", type=str,
    help="checkpoint for grounding dino"
)
parser.add_argument(
    '--sam_checkpoint', required=False,
    default="./checkpoints/sam2.1_hiera_large.pt", type=str,
    help="checkpoint for sam"
)
parser.add_argument(
    '--sam_yaml', required=False,
    default="configs/sam2.1/sam2.1_hiera_l.yaml", type=str,
    help="yaml config for sam"
)
args = parser.parse_args()

data_dir = os.path.join(args.input_dir, "train_0/camera")
frames_dir = os.path.join(data_dir, "rgb")

mask_dir = os.path.join(data_dir, "color_mask")
bmask_dir = os.path.join(data_dir, "binary_mask")
class_dir = os.path.join(data_dir, "class")
# bmask_each_dir = os.path.join(data_dir, "binary_mask_each")
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(bmask_dir, exist_ok=True)
os.makedirs(class_dir, exist_ok=True)
# os.makedirs(bmask_each_dir, exist_ok=True)

###############################
# Load model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.2
TEXT_PROMPT = "786dvpteg. object."

grounding_model = load_model(
    model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
    model_checkpoint_path=args.gd_checkpoint,
    device=DEVICE
)
video_predictor = build_sam2_video_predictor(args.sam_yaml, args.sam_checkpoint)
sam2_image_model = build_sam2(args.sam_yaml, args.sam_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)
frame_names = [
    p for p in os.listdir(frames_dir)
    if os.path.splitext(p)[-1] in [".JPG", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))





GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "checkpoint.pth"
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
start = tm.time()
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
# Using mask prompt is a more straightforward way
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

video_segments = {}  # video_segments contains the per-frame segmentation results
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
bounding_boxes = []
image_paths = []

reindex = 0
index_list = []
num_obj = len(masks)
for frame_idx, segments in video_segments.items():
    save_path = os.path.join(save_folder_mask, f'{reindex:06d}.png')
    save_path_binary = os.path.join(save_binary_mask_folder, f'{reindex:06d}.png')
    save_path_class = os.path.join(save_class_folder, f'{reindex:06d}.png')
    save_bbox_path = os.path.join(save_bbox, f'{reindex:06d}.npy')
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

# def mat2dict(mat, w , h):
#     fl_x=mat[0,0]
#     fl_y= mat[1,1]
#     camera_angle_x = math.atan(w / (fl_x * 2)) * 2
#     camera_angle_y = math.atan(h / (fl_y * 2)) * 2
#     cx=mat[0,2]
#     cy= mat[1,2]
#     cam = {		"camera_angle_x": camera_angle_x,
#                 "camera_angle_y": camera_angle_y,
#                 "fl_x": fl_x,
#                 "fl_y": fl_y,
#                 "cx": cx,
#                 "cy": cy,
#                 "w": w,
#                 "h": h,
#             }
#     return cam

# hand_eye_init = np.array([[-0.00655526, -0.999961  ,  0.00593217,  0.092329  ],
#                             [ 0.999813  , -0.006662  , -0.0181578 ,  0.00308126],
#                             [ 0.0181966 ,  0.00581203,  0.999818  ,  0.0622155 ],
#                             [ 0.        ,  0.        ,  0.        ,  1.        ]])
# out = {		
#             "num_obj": num_obj,
#             "hand_eye_init": hand_eye_init.tolist(),
# 			"frames": [],
# 		}

# out_ = {		
#             "num_obj": num_obj,
#             "hand_eye_init": hand_eye_init.tolist(),
# 			"frames": [],
# 		}


# seq_index = f"real_test_{opt.index:02d}"
# dynamic_cam_image_folder = os.path.join(f"data/{seq_index}","train_0/camera/rgb")
# dynamic_cam_mask_folder = os.path.join(f"data/{seq_index}","train_0/camera/mask_")
# dynamic_cam_binary_mask_folder = os.path.join(f"data/{seq_index}","train_0/camera/binary_mask")
# dynamic_cam_class_folder = os.path.join(f"data/{seq_index}","train_0/camera/class")
# dynamic_cam_depth = os.path.join(f"data/{seq_index}","train_0/camera/depth")
# pose_folder = os.path.join(f"data/{seq_index}","train_0/pose")
# dynamic_image_files = [f for f in sorted(os.listdir(dynamic_cam_mask_folder)) if f.endswith(".png")]

# for idx, image_file in enumerate(dynamic_image_files):
#     image_name = image_file.split('.')[0]
#     pose_idx = idx+1
#     image_path = os.path.join(dynamic_cam_mask_folder,image_name+".png") 
#     original_image_path = os.path.join(dynamic_cam_image_folder,f"{idx+1:05d}"+".JPG") 
#     binay_mask_path = os.path.join(dynamic_cam_binary_mask_folder,image_name+".png") 
#     class_path = os.path.join(dynamic_cam_class_folder,image_name+".png")
#     pose_idx = idx+1
#     pose_txt = os.path.join(pose_folder, f"{pose_idx:05d}"+".txt")
#     depth = os.path.join(dynamic_cam_depth,f"{pose_idx:05d}"+".png")
#     with open(pose_txt) as f:
#         cam_pose = f.read()
#     matrix_values = np.array([float(i) for i in cam_pose.split()]).reshape(4,4)
#     pose_list = matrix_values.tolist()
#     cam = mat2dict(K, W, H)
#     new_frame = {
#     "file_path": f"{image_path}",
#     "original_path": f"{original_image_path}",
#     "binary_mask_path": f"{binay_mask_path}",
#     "class_path": f"{class_path}",
#     "transform_matrix": pose_list,
#     "depth_path": f"{depth}",
#     "cam": cam,
#     "obj_idx": index_list,
#     }
#     if idx %4 ==0:
#         out["frames"].append(new_frame)
#     else:
#         out_["frames"].append(new_frame)

# with open(os.path.join(f"data/{seq_index}","transforms_sparse_train.json"), 'w') as f:
#    json.dump(out, f, indent="\t")
# with open(os.path.join(f"data/{seq_index}","transforms_sparse_test.json"), 'w') as f:
#    json.dump(out_, f, indent="\t")

