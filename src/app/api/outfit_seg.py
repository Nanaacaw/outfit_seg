# app/api/outfit_seg.py

from fastapi import APIRouter
from app.schemas.segmentation import LabelRequest
from app.services.segmentation_service import grounded_segmentation
from app.utils.plotting import plot_detections
import os
from datetime import datetime
from app.settings.setting import DEFAULT_THRESHOLD, DETECTOR_ID, SEGMENTER_ID
from PIL import Image


router = APIRouter()

@router.post("/segment")
def segment_outfit(request: LabelRequest):
    try:
        # Use threshold from request, if None use from settings
        threshold = request.threshold if request.threshold is not None else DEFAULT_THRESHOLD
        polygon_refinement = request.polygon_refinement if request.polygon_refinement is not None else True

        image_array, detections = grounded_segmentation(
            image=request.image_url,
            labels=request.labels,
            threshold=threshold,
            polygon_refinement=polygon_refinement,
            detector_id=DETECTOR_ID,
            segmenter_id=SEGMENTER_ID
        )
        
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/segmentation_{timestamp}.png"

        # Convert numpy array to PIL Image
        image_pil = Image.fromarray(image_array)
        plot_detections(image_pil, detections, save_name=filename)

        # Save segmentation result...
        return {
            "status": "completed",
            "data": {
                "image_url": request.image_url,
                "num_detections": len(detections),
                "saved_file": filename
            }
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