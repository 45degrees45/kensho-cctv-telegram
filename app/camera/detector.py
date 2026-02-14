from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import YOLO_MODEL, YOLO_CONFIDENCE, YOLO_CLASSES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO class ID -> name mapping for the classes we care about.
# ---------------------------------------------------------------------------
COCO_CLASS_IDS: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    21: "bear",
}

# Reverse lookup: name -> COCO ID.
COCO_NAME_TO_ID: dict[str, int] = {v: k for k, v in COCO_CLASS_IDS.items()}

# Colour palette for bounding boxes by category.
#   Red   = person
#   Blue  = vehicle (car, truck, motorcycle, bicycle, bus)
#   Green = animal  (cat, dog, bird, horse, cow, sheep, bear)
_VEHICLES = {"car", "truck", "motorcycle", "bicycle", "bus"}
_ANIMALS = {"cat", "dog", "bird", "horse", "cow", "sheep", "bear"}

_COLOR_PERSON = (0, 0, 255)    # BGR red
_COLOR_VEHICLE = (255, 0, 0)   # BGR blue
_COLOR_ANIMAL = (0, 200, 0)    # BGR green


def _box_color(label: str) -> tuple[int, int, int]:
    if label == "person":
        return _COLOR_PERSON
    if label in _VEHICLES:
        return _COLOR_VEHICLE
    if label in _ANIMALS:
        return _COLOR_ANIMAL
    return (255, 255, 255)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class YOLODetector:
    """Wraps ultralytics YOLO for detecting people, vehicles, and animals."""

    def __init__(self) -> None:
        # Build the set of allowed COCO class IDs from the config string.
        allowed_names = {n.strip().lower() for n in YOLO_CLASSES.split(",")}
        self._allowed_ids: list[int] = [
            COCO_NAME_TO_ID[n] for n in allowed_names if n in COCO_NAME_TO_ID
        ]

        logger.info("Loading YOLO model: %s", YOLO_MODEL)
        self._model = YOLO(YOLO_MODEL)
        logger.info(
            "YOLO ready — confidence=%.2f, classes=%s",
            YOLO_CONFIDENCE,
            sorted(allowed_names & set(COCO_CLASS_IDS.values())),
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO inference on a BGR frame.

        Returns a list of Detection objects for accepted classes above
        the confidence threshold.
        """
        results = self._model(
            frame,
            conf=YOLO_CONFIDENCE,
            classes=self._allowed_ids,
            verbose=False,
        )

        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = COCO_CLASS_IDS.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                ))

        return detections

    @staticmethod
    def annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame."""
        annotated = frame.copy()

        for det in detections:
            color = _box_color(det.label)
            x1, y1, x2, y2 = det.bbox

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            text = f"{det.label} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated, text, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        return annotated

    @staticmethod
    def build_caption(detections: list[Detection]) -> str:
        """Build a human-readable caption like 'Detected: 2x person, 1x car'."""
        counts = Counter(d.label for d in detections)
        parts = [f"{count}x {label}" for label, count in counts.most_common()]
        return f"Detected: {', '.join(parts)}"
