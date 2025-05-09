import time
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple

import cv2
import imutils
from tqdm import tqdm

from report.dataset import parse_fddb_annotations
from report.project_typing import BoundingBox


def compute_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea != 0 else 0


def compute_recall(
    predicted_boxes: list[BoundingBox], true_boxes: list[BoundingBox], iou_threshold=0.5
) -> float:
    matched = set()
    for true_box in true_boxes:
        for pred_box in predicted_boxes:
            if (
                compute_iou(pred_box, true_box) >= iou_threshold
                and true_box not in matched
            ):
                matched.add(true_box)
                break
    TP = len(matched)
    FN = len(true_boxes) - TP
    return TP / (TP + FN) if (TP + FN) > 0 else 0


def compute_precision(
    predicted_boxes: list[BoundingBox], true_boxes: list[BoundingBox], iou_threshold=0.5
) -> float:
    matched = set()
    for pred_box in predicted_boxes:
        for true_box in true_boxes:
            if (
                compute_iou(pred_box, true_box) >= iou_threshold
                and true_box not in matched
            ):
                matched.add(true_box)
                break
    TP = len(matched)
    FP = len(predicted_boxes) - TP
    return TP / (TP + FP) if (TP + FP) > 0 else 0


def evaluate_detector(
    detector_fn: Callable[[Any], list[BoundingBox]],
    image_modification: Callable[[Any], Any] | None = None,
    width: int = 600,
    limit: int = 1_000,
    annotation_file="fddb/FDDB-folds/FDDB-fold-01-ellipseList.txt",
    image_base="fddb/images",
) -> tuple[float, float, float, float]:
    dataset = parse_fddb_annotations(annotation_file, image_base, limit=limit)

    total_pixels = 0
    total_time = 0.0
    total_recall = 0.0
    total_precision = 0.0

    progress_bar = tqdm(dataset, desc="Evaluating", unit="img")

    for image_path, true_boxes in progress_bar:
        img = cv2.imread(image_path)
        if img is None:
            continue  # skip corrupted or missing images

        if image_modification:
            img = image_modification(img)

        img, true_boxes = scale_by_width(img, true_boxes, width)

        height, width = img.shape[:2]
        total_pixels += height * width

        start = time.perf_counter()
        predicted_boxes = detector_fn(img)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        recall = compute_recall(predicted_boxes, true_boxes)
        precision = compute_precision(predicted_boxes, true_boxes)

        total_recall += recall
        total_precision += precision

        progress_bar.set_postfix_str(f"{recall:.2f};{precision:.2f}")

    n = len(dataset)
    avg_speed = total_time / (total_pixels / 1_000_000)  # seconds per megapixel
    avg_recall = total_recall / n
    avg_precision = total_precision / n

    avg_total_megapixels = (total_pixels / 1_000_000) / limit
    return avg_speed, avg_recall, avg_precision, avg_total_megapixels


import cv2
from typing import Callable


def visualize_detector(
    face_detector: Callable[[Any], list[BoundingBox]],
    image_modification: Callable[[Any], Any] | None = None,
    annotation_file: str = "wider_face_split/wider_face_val_bbx_gt.txt",
    image_base: str = "WIDER_val/images",
    limit: int = 10,
    width: int = 600,
):
    dataset = parse_fddb_annotations(annotation_file, image_base, limit=limit)

    for image_path, true_boxes in dataset:
        img = cv2.imread(image_path)
        img, true_boxes = scale_by_width(img, true_boxes, width)

        if img is None:
            continue

        if image_modification is not None:
            img = image_modification(img)

        pred_boxes = face_detector(img)

        # Draw ground truth boxes (green)
        for x, y, w, h in true_boxes:
            cv2.rectangle(
                img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
            )

        # Draw predicted boxes (red)
        for x, y, w, h in pred_boxes:
            cv2.rectangle(
                img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2
            )

        # Display
        cv2.imshow("Face Detection - Green: GT | Red: Prediction", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def scale_by_width(
    img, true_boxes: List[BoundingBox], width: int
) -> Tuple[Any, List[BoundingBox]]:
    h_orig, w_orig = img.shape[:2]

    # Resize the image
    img_resized = imutils.resize(img, width=width)
    h_new, w_new = img_resized.shape[:2]

    # Compute scale factors
    scale_x = w_new / w_orig
    scale_y = h_new / h_orig

    # Scale bounding boxes
    scaled_boxes = [
        (x * scale_x, y * scale_y, w * scale_x, h * scale_y)
        for (x, y, w, h) in true_boxes
    ]

    return img_resized, scaled_boxes
