import os

import cv2


def ellipse_to_bbox(major_axis, minor_axis, angle, center_x, center_y):
    # For simplicity, we approximate the ellipse as a bounding box centered at (cx, cy)
    # covering the ellipse fully regardless of rotation
    w = 2 * major_axis
    h = 2 * minor_axis
    x = center_x - w / 2
    y = center_y - h / 2
    return (x, y, w, h)


def parse_fddb_annotations(
    annotation_file: str, base_image_path: str, limit: int = 100
):
    dataset = []
    with open(annotation_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines) and len(dataset) < limit:
        filename = lines[i] + ".jpg"  # FDDB filenames omit extension
        full_path = os.path.join(base_image_path, filename)
        i += 1

        face_count = int(lines[i])
        i += 1

        boxes = []
        for _ in range(face_count):
            major_axis, minor_axis, angle, cx, cy, _ = map(float, lines[i].split())
            bbox = ellipse_to_bbox(major_axis, minor_axis, angle, cx, cy)
            boxes.append(bbox)
            i += 1

        dataset.append((full_path, boxes))

    return dataset
