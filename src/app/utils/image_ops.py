# app/utils/image_ops.py

from PIL import Image
import torch
import numpy as np
import cv2
from typing import List, Tuple

from .results import DetectionResult
from .s3_helper import download_from_s3

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon

def polygon_to_mask(polygon: List[List[int]], image_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http") or image_str.startswith("s3://"):
        local_path = download_from_s3(image_str)
        image = Image.open(local_path).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    return image

def get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        boxes.append(result.box.xyxy)
    return [boxes]

def refine_masks(masks: torch.Tensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(dim=-1)
    masks = (masks > 0).int()
    masks_np = masks.numpy().astype(np.uint8)
    masks_list = list(masks_np)

    if polygon_refinement:
        for idx, mask in enumerate(masks_list):
            shape = tuple(mask.shape)
            polygon = mask_to_polygon(mask)
            refined_mask = polygon_to_mask(polygon, shape)
            masks_list[idx] = refined_mask

    return masks_list
