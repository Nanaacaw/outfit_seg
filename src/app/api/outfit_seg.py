# app/api/outfit_seg.py

from fastapi import APIRouter, UploadFile, File, Form
from app.schemas.segmentation import LabelRequest
from app.services.segmentation_service import grounded_segmentation
from app.utils.plotting import plot_detections
import os, io, torch, platform, psutil
from datetime import datetime
from app.settings.setting import DEFAULT_THRESHOLD, DETECTOR_ID, SEGMENTER_ID
from PIL import Image
from typing import List, Optional


router = APIRouter()

DEFAULT_LABELS = ["person", "shirt", "pant", "shoe", "sandal", "headscarf", "watch", "glasses", "skirt", "vest", "hat"]

@router.post("/segment")
def segment_outfit(request: LabelRequest):
    try:
        # Use threshold from request, if None use from settings
        threshold = request.threshold if request.threshold is not None else DEFAULT_THRESHOLD
        polygon_refinement = request.polygon_refinement if request.polygon_refinement is not None else True

        # Use labels from request if not None, otherwise use default labels
        labels = request.labels if request.labels is not None else DEFAULT_LABELS
        labels = [label if label.endswith(".") else label + "." for label in labels]

        image_array, detections = grounded_segmentation(
            image=request.image_url,
            labels=labels,
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
@router.post("/segment-local")
async def segment_local(
    file: UploadFile = File(...), 
    labels: Optional[List[str]] = None,
    threshold: float = Form(DEFAULT_THRESHOLD),
    polygon_refinement: Optional[bool] = Form(None)
):
    try:
        # Read content of the file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Use labels from request if not None, otherwise use default labels
        labels = labels if labels is not None else DEFAULT_LABELS
        labels = [label if label.endswith(".") else label + "." for label in labels]

        # Use polygon refinement from request if not None, otherwise use True   
        polygon_refinement = polygon_refinement if polygon_refinement is not None else True

        image_array, detections = grounded_segmentation(
            image=image,
            labels=labels,
            threshold=threshold,
            polygon_refinement=polygon_refinement,
            detector_id=DETECTOR_ID,
            segmenter_id=SEGMENTER_ID
        )

        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/segmentation_local_{timestamp}.png"

        image_pil = Image.fromarray(image_array)
        plot_detections(image_pil, detections, save_name=filename)

        return {
            "status": "completed",
            "data": {
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

@router.get("/status")
def check_status():
    try:
        # Check cuda
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None

        return {
            "status": "ok",
            "cuda_available": cuda_available,
            "device_name": cuda_device if cuda_available else "CPU",
            "torch_version": torch.__version__,
            "platform": platform.system(),
            "memory_usage_percent": psutil.virtual_memory().percent
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }