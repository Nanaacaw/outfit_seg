# app/services/segmentation_service.py

from typing import List, Optional, Tuple, Union
from PIL import Image
import numpy as np

from app.core.detect import detect
from app.core.segment import segment
from app.utils.image_ops import load_image
from app.utils.results import DetectionResult
from app.settings.setting import DEFAULT_THRESHOLD, DETECTOR_ID, SEGMENTER_ID

def grounded_segmentation(
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
        image = load_image(image)

    detections = detect(
        image=image,
        labels=labels,
        threshold=threshold,
        detector_id=detector_id
    )

    detections = segment(
        image=image,
        detection_results=detections,
        polygon_refinement=polygon_refinement,
        segmenter_id=segmenter_id
    )

    return np.array(image), detections
