from typing import List
from app.utils.results import DetectionResult

# app/services/detection_filter.py
def compute_iou(boxA: list, boxB: list) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Each box is [xmin, ymin, xmax, ymax].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0.0
    return interArea / union

def remove_multilabel_same_area(
    detections: List[DetectionResult], iou_threshold: float = 0.4
) -> List[DetectionResult]:
    """
    Untuk setiap area overlap tinggi (IoU > threshold), hanya simpan deteksi dengan skor tertinggi (apapun labelnya).
    """
    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    kept = []
    used = set()

    for i, det in enumerate(detections):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(detections)):
            if j in used:
                continue
            iou = compute_iou(det.box.xyxy, detections[j].box.xyxy)
            if iou > iou_threshold:
                group.append(j)
        # Pilih deteksi dengan skor tertinggi di grup (sudah urut, jadi ambil deteksi pertama)
        kept.append(det)
        # Tandai semua di grup sebagai sudah diproses
        for idx in group:
            used.add(idx)
    return kept

