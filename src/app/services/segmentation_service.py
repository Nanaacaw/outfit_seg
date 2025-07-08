# app/services/segmentation_service.py

from typing import List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import asyncio

from src.app.core.detect import detect
from src.app.core.segment import segment
from app.utils.image_ops import load_image
from app.utils.results import DetectionResult
from app.settings.setting import DEFAULT_THRESHOLD, DETECTOR_ID, SEGMENTER_ID

async def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = DEFAULT_THRESHOLD,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    """
    Pipeline: Load image, detect objects, and segment masks.
    """
    if isinstance(image, str):
        image_pil = await load_image(image)
    else:
        image_pil = image

    # Run detection and segmentation concurrently
    detections = await detect(
        image=image_pil,
        labels=labels,
        threshold=threshold,
        detector_id=detector_id
    )

    detections = await segment(
        image=image_pil,
        detection_results=detections,
        polygon_refinement=polygon_refinement,
        segmenter_id=segmenter_id
    )

    return np.array(image), detections
