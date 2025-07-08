# app/core/segment.py

from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from transformers import AutoModelForMaskGeneration, AutoProcessor

from app.utils.image_ops import refine_masks
from app.utils.image_ops import get_boxes
from app.utils.results import DetectionResult
from app.settings.setting import SEGMENTER_ID

def segment(
    image: Image.Image,
    detection_results: List[DetectionResult],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = segmenter_id if segmenter_id is not None else SEGMENTER_ID

    segmentator = AutoModelForMaskGeneration.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    boxes = get_boxes(detection_results)
    inputs = processor(
        images=image,
        input_boxes=boxes,
        return_tensors="pt"
    ).to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results
