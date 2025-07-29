import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from minlora import add_lora, LoRAParametrization, get_lora_params
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
import torch.nn as nn
from functools import partial
from PIL import Image
def parse_arguments():
    parser = argparse.ArgumentParser(description="CLIP training with object captions.")
    parser.add_argument('-o.', "--obj_num", type=int, required=True, help="Number of objects per image.")
    parser.add_argument('-i', '--index', required=True,
                    default=0, type=int,
                    help="index")  
    
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for training.")
    parser.add_argument('--lora_rank', type=int, default=64)
    return parser.parse_args()

def crop_and_preprocess(image_path, boxes, transform):
    image = Image.open(image_path).convert("RGB")
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cropped = image.crop((x1, y1, x2, y2))  # Bounding Box로 crop
        crops.append(transform(cropped))
    return torch.stack(crops)

def find_caption_for_bboxes(model, processor, image_path, bboxes, captions, device):
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    cropped_images = crop_and_preprocess(image_path, bboxes, transform).to(device)

    # 텍스트 임베딩 계산
    with torch.no_grad():
        print(captions)
        text_inputs = processor.tokenizer(captions, return_tensors="pt", padding=True).to(device)
        text_embeddings = model.get_text_features(**text_inputs)  # (num_captions, embed_dim)

        # 이미지 임베딩 계산
        image_embeddings = []
        for cropped_image in cropped_images:
            cropped_image = cropped_image.unsqueeze(0)  # Add batch dimension
            image_inputs = processor.feature_extractor(images=cropped_image, return_tensors="pt", do_rescale=False).to(device)
            image_embedding = model.get_image_features(**image_inputs)  # (1, embed_dim)
            image_embeddings.append(image_embedding)

        image_embeddings = torch.cat(image_embeddings, dim=0)  # (num_bboxes, embed_dim)

        # 유사도 계산
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.T)  # (num_bboxes, num_captions)

        # 가장 유사한 캡션 인덱스 찾기
        best_caption_indices = torch.argmax(logits_per_image, dim=1)  # (num_bboxes,)
        best_captions = [captions[idx] for idx in best_caption_indices.cpu().numpy()]

    return best_captions


args = parse_arguments()
obj_num = args.obj_num
device = "cuda" if torch.cuda.is_available() else "cpu"
seq_index = f"multi_test_{args.index:02d}"
test_path =  f"/mydata/jyk/ICCV2025/{seq_index}"
checkpoint_path = os.path.join(test_path,"clip_epoch_40.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
version = "openai/clip-vit-large-patch14" # "openai/clip-vit-base-patch32"
# version = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(version)
lora_config = {
    nn.Embedding: {
        "weight": partial(LoRAParametrization.from_embedding, rank=args.lora_rank)
    },
    nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=args.lora_rank)
    },
    nn.Conv2d: {
        "weight": partial(LoRAParametrization.from_conv2d, rank=args.lora_rank)
    }
}


add_lora(model, lora_config=lora_config)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
processor = CLIPProcessor.from_pretrained(version)
mobilesamv2, ObjAwareModel, predictor_mobile_sam = torch.hub.load("RogerQi/MobileSAMV2", "mobilesamv2_efficientvit_l2")
mobilesamv2.to(device=device)
mobilesamv2.eval()
for param in model.parameters():
    param.requires_grad = False

model.eval()
model.to(device)
object_ids = [f"obj{i+1}" for i in range(obj_num)]
import pickle
with open(os.path.join(test_path, 'caption.pkl'), 'rb') as f:
    object_id_to_caption = pickle.load(f)
print(object_id_to_caption)
static_cam_image_folder = os.path.join(test_path,"camera/D/crop")
IMG_PATH = os.path.join(static_cam_image_folder,"00017.JPG") 

TEXT_PROMPT = "glass. plastic."
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.29
TEXT_THRESHOLD = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)


def add_offset_to_bbox(bbox, offset=30, img_width=1280, img_height=720):
    x, y, width, height = bbox
    x_new = max(0, x - offset)
    y_new = max(0, y - offset)
    width_new = min(img_width - x_new, width + 2 * offset)
    height_new = min(img_height - y_new, height + 2 * offset)
    return [x_new, y_new, width_new, height_new]


text = TEXT_PROMPT
img_path = IMG_PATH
image_source, image = load_image(img_path)
sam2_predictor.set_image(image_source)

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=text,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)
img_ = cv2.imread(img_path)
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


captions = [object_id_to_caption[obj_id] for obj_id in object_ids]
results = find_caption_for_bboxes(model, processor, img_path, input_boxes, captions, device)
print("Bounding Box to Caption Mapping:")
for bbox, caption in zip(input_boxes, results):
    print(f"BBox {bbox} -> {caption}")
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

# obj_mask = masks[-1, :,:]
# nonzero = np.where(obj_mask>0)
# y_min, x_min = np.min(nonzero, axis=1)
# y_max, x_max = np.max(nonzero, axis=1)

# x_min, y_min, x_max, y_max = input_boxes[obj_index]

# crop_image = image_source[y_min:y_max,x_min:x_max]

# cropped_images = crop_mask(masks, image_source)
# print(cropped_images.shape, masks.shape, image_source.shape)
# cv2.imwrite(os.path.join(OUTPUT_DIR, "crop.JPG"), crop_image)
confidences = confidences.numpy().tolist()
class_names = labels

class_ids = np.array(list(range(len(class_names))))

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

"""
Visualize image with supervision useful API
"""
img = cv2.imread(img_path)
detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks.astype(bool),  # (n, h, w)
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "sibal.jpg"), annotated_frame)

"""
Dump the results in standard format and save as json files
"""

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    # save the results in standard format
    results = {
        "image_path": img_path,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }
    
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)