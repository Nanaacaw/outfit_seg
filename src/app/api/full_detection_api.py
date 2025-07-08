# app/api/full_detection.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from typing import Optional
from datetime import datetime
from PIL import Image
import os, json, torch, platform, psutil, numpy as np
from io import BytesIO
import logging
from src.app.utils.logger_utils import get_logger, debug_log

logger = get_logger(__name__)

from app.services.segmentation_service import grounded_segmentation
from app.utils.plotting import plot_detections
from app.settings.setting import DEFAULT_THRESHOLD, DETECTOR_ID, SEGMENTER_ID
from app.utils.image_ops import load_image
from app.utils.postprocess import remove_multilabel_same_area, compute_iou

router = APIRouter()

DEFAULT_LABELS = ["person.", "shirt.", "pant.", "shoe.", "sandal.", "headscarf.", "watch.", "glasses.", "skirt.", "vest.", "hat."]

@router.post("/detect")
async def detect (
    image_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    labels: Optional[str] = Form(None),
    threshold: Optional[float] = Form(DEFAULT_THRESHOLD),
    polygon_refinement: Optional[bool] = Form(True)
):
    try:
        debug_log("/detect request received", logger)
        # Handle labels
        label_list = labels.split(",") if labels else DEFAULT_LABELS
        if not any(l.lower() == "person" for l in label_list):
            label_list = ["person"] + label_list
        label_list = [label if label.endswith(".") else label + "." for label in label_list]

        # Load image
        if image_url:
            image_pil = load_image(image_url)
            input_type = "url"
            image_source = image_url
        elif file:
            image_pil = Image.open(BytesIO(await file.read())).convert("RGB")
            input_type = "file"
            image_source = file.filename
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        image_array = np.array(image_pil)

        # Handle threshold and polygon refinement
        threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        polygon_refinement = polygon_refinement if polygon_refinement is not None else True

        debug_log(f"Detection started for image: {image_source}", logger)
        image_array, detections = grounded_segmentation(
            image=image_pil,
            labels=label_list,
            threshold=threshold,
            polygon_refinement=polygon_refinement,
            detector_id=DETECTOR_ID,
            segmenter_id=SEGMENTER_ID
        )
        debug_log(f"Detection completed: {len(detections)} detections", logger)

        # Filter by score threshold
        detections = [d for d in detections if d.score >= threshold]

        # Separate detections into persons and outfit items
        persons = [d for d in detections if d.label.lower().strip().rstrip('.') == 'person']
        items = [d for d in detections if d.label.lower().strip().rstrip('.') != 'person']

        # Filter overlapping items (only keep highest score per area)
        items = remove_multilabel_same_area(items, iou_threshold=0.5)

        # Get image size
        img_width, img_height = image_pil.size

        def normalize_box(xmin, ymin, xmax, ymax):
            x = xmin / img_width
            y = ymin / img_height
            w = (xmax - xmin) / img_width
            h = (ymax - ymin) / img_height
            return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]

        # Build results
        results = []
        for idx, person in enumerate(persons, 1):
            pxmin, pymin, pxmax, pymax = person.box.xyxy
            outfits = []
            for item in items:
                ixmin, iymin, ixmax, iymax = item.box.xyxy
                if (ixmin >= pxmin and iymin >= pymin and ixmax <= pxmax and iymax <= pymax):
                    outfits.append({
                        "text_prompt": item.label,
                        "box": normalize_box(ixmin, iymin, ixmax, iymax),
                        "confidence": round(item.score, 4),
                        "iou_with_person": round(compute_iou([pxmin, pymin, pxmax, pymax], [ixmin, iymin, ixmax, iymax]), 4)
                    })

            results.append({
                "person_id": idx,
                "bounding_box": normalize_box(pxmin, pymin, pxmax, pymax),
                "outfit": outfits
            })

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        image_filename = f"{results_dir}/detection_{timestamp}.png"
        json_filename = f"{results_dir}/detection_{timestamp}.json"
        plot_detections(image_pil, detections, save_name=image_filename)

        response = {
            "input_type": input_type,
            "image_source": image_source,
            "status": "completed",
            "num_persons": len(persons),
            "total_detections": len(detections),
            "results": results,
            "saved_files": {
                "image": image_filename,
                "json": json_filename
            }
        }

        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2)

        return response
    except Exception as e:
        logger.error("Detection failed: %s", str(e))
        return {
            "input_type": "url",
            "status": "failed",
            "error": str(e)
        }

@router.get("/results")
def get_specific_result(filename: str = Query(..., description="Filename of the image result to fetch")):
    results_dir = "results"
    
    if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".webp")):
        return {"error": "Only .png, .jpg, .webp image filenames are allowed."}
    
    image_path = os.path.join(results_dir, filename)
    json_path = os.path.splitext(image_path)[0] + ".json"

    if not os.path.exists(image_path) or not os.path.exists(json_path):
        return {"error": "File not found."}

    with open(json_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    return {
        "filename": filename,
        "image_url": f"/results/{filename}",
        "result": result_data
    }

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