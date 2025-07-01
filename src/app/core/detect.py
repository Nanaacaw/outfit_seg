# app/core/detect_utils.py

from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from transformers import pipeline

from app.utils.results import DetectionResult
from app.settings.setting import DETECTOR_ID, DEFAULT_THRESHOLD

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = DEFAULT_THRESHOLD,
    detector_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = detector_id if detector_id is not None else DETECTOR_ID

    object_detector = pipeline(
        model=model_id,
        task="zero-shot-object-detection",
        device=device
    )

    labels = [label if label.endswith(".") else label + "." for label in labels]

    raw_results = object_detector(
        image,
        candidate_labels=labels,
        threshold=threshold
    )

    results = [DetectionResult.from_dict(r) for r in raw_results]

    return results
