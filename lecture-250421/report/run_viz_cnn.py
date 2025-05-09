from report.evaluation import visualize_detector
from report.face_detectors import cnn_face_detector
from report.shared_config import widths

annotation_file = "fddb/FDDB-folds/FDDB-fold-01-ellipseList.txt"
image_base = "fddb/images"

for width in [300]:
    visualize_detector(
        cnn_face_detector,
        image_modification=None,
        annotation_file=annotation_file,
        image_base=image_base,
        width=width,
        limit=10,  # only 10
    )
