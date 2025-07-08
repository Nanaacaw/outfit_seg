# app/core/detect.py

from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from transformers.pipelines import pipeline
import logging
from src.app.utils.logger_utils import get_logger, debug_log

logger = get_logger(__name__)

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
    try:
        debug_log("Detection started", logger)
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
        debug_log(f"Detection completed: {len(results)} results", logger)
        return results
    except Exception as e:
        logger.error("Detection error: %s", str(e))
        raise
