import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import argparse
import random
import json
import math
import time as tm
import gc

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
parser.add_argument(
    '--dataset', required=False,
    default="syn", type=str,
    help="dataset"
)
args = parser.parse_args()

if args.dataset == "syn":
    data_dir = os.path.join(args.input_dir, f"train_0/data")
    frames_dir = os.path.join(data_dir, "RGB")
elif args.dataset == "real":
    data_dir = os.path.join(args.input_dir, "train_0/camera")
    frames_dir = os.path.join(data_dir, "rgb")
else:
    raise ValueError("dataset should be 'syn' or 'real'")

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

###############################
# Grounding DINO
inference_state = video_predictor.init_state(video_path=frames_dir)
ann_frame_idx = 0
img_path = os.path.join(frames_dir, frame_names[ann_frame_idx])
image_source, image = load_image(img_path)
boxes, confidences, class_names = predict(
    model=grounding_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    remove_combined = True
)

del grounding_model
torch.cuda.empty_cache()
torch.cuda.synchronize()
gc.collect()

###############################
# SAM2
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
image_predictor.set_image(image_source)

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
for object_id, (label, mask) in enumerate(zip(class_names, masks), start=1):
    _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=object_id,
        mask=mask
    )

video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

reindex = 0
index_list = []
num_obj = len(masks)
for frame_idx, segments in video_segments.items():
    save_path_mask = os.path.join(mask_dir, f'{reindex:06d}.png')
    save_path_binary = os.path.join(bmask_dir, f'{reindex:06d}.png')
    save_path_class = os.path.join(class_dir, f'{reindex:06d}.png')
    img = cv2.imread(os.path.join(frames_dir, frame_names[frame_idx]))
    H,W,C = img.shape
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    class_image = np.zeros((H, W), dtype=np.uint8)
    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)
    class_id=np.array(object_ids, dtype=np.int32)
    for i in range(len(masks)):
        binary_mask_each = np.zeros((H, W), dtype=np.uint8)
        mask_ = np.stack((masks[i]>0, masks[i]>0, masks[i]>0), axis = -1)
        binary_mask[masks[i]!=0] = 255
        binary_mask_each[masks[i]!=0] = 255
        class_image[masks[i]!=0] = (int(class_id[i]))
        # save_path_binary_each = os.path.join(bmask_each_dir, f'{reindex:06d}_{i:06d}.png')
        # cv2.imwrite(save_path_binary_each, binary_mask_each)
        color = colors[class_id[i]]
        mask_image = masks[i].reshape(h, w, 1)* color.reshape(1, 1, -1)
        img = img*(1-mask_) + mask_image[:,:,:3]*mask_*255.0*0.7 + img[:,:,:3]*mask_*0.3
        if frame_idx==0:
            index_list.append(i+1)
    cv2.imwrite(save_path_binary, binary_mask)
    cv2.imwrite(save_path_class, class_image)
    cv2.imwrite(save_path_mask, img)
    reindex+=1

# save index_list
with open(os.path.join(data_dir, "obj_idx.json"), 'w') as f:
    json.dump(index_list, f, indent="\t")