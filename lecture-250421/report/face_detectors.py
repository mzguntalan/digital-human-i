import argparse
import time
from typing import Any

import cv2
import dlib
import imutils

from pyimagesearch.helpers import convert_and_trim_bb
from report.project_typing import BoundingBox


def cnn_face_detector(image: Any) -> list[BoundingBox]:
    detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = detector(rgb, 1)
    boxes = [convert_and_trim_bb(image, r.rect) for r in results]

    return boxes


def hog_face_detection(image: Any) -> list[BoundingBox]:
    detector = dlib.get_frontal_face_detector()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = detector(rgb, 1)
    boxes = [convert_and_trim_bb(image, r) for r in results]

    return boxes
