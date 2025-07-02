# app/api/outfit_seg.py

from fastapi import APIRouter
from app.schemas.segmentation import LabelRequest
from app.services.segmentation_service import grounded_segmentation
from app.utils.plotting import plot_detections
import os
from datetime import datetime
from app.settings.setting import *


router = APIRouter()

@router.post("/segment")
def segment_outfit(request: LabelRequest):
    try:
        # Gunakan threshold dari request, kalau None pakai dari settings
        threshold = request.threshold if request.threshold is not None else DEFAULT_THRESHOLD

        image_array, detections = grounded_segmentation(
            image=request.image_url,
            labels=request.labels,
            threshold=threshold,
            polygon_refinement=request.polygon_refinement,
            detector_id=DETECTOR_ID,
            segmenter_id=SEGMENTER_ID
        )
        
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/segmentation_{timestamp}.png"

        plot_detections(image_array, detections, save_name=filename)

        # save segmentation result...
        return {
            "image_url": request.image_url,
            "status": "completed",
            "num_detections": len(detections),
            "saved_file": filename
        }
    except Exception as e:
        return {
            "status": "failed",
            "num_detections": 0,
            "error": str(e)
        }

@router.get("/results")
def list_segmentation_results():
    results_dir = "results"
    if not os.path.exists(results_dir):
        return {"files": []}

    files = sorted(os.listdir(results_dir))
    file_urls = [
        {
            "filename": file,
            "url": f"/results/{file}"
        }
        for file in files if file.endswith(".png")
    ]
    return {"files": file_urls}