from report.evaluation import evaluate_detector
from report.face_detectors import hog_face_detection
from report.shared_config import limit_number_of_images
from report.shared_config import widths


print("HOG Face Detector")
for width in widths:
    print(f"Width: {width}")
    results = evaluate_detector(
        hog_face_detection,
        None,
        width,
        limit_number_of_images,
    )
    print(f"\tSpeed (s/MP):", results[0])
    print(f"\tRecall:", results[1])
    print(f"\tPrecision:", results[2])
    print(f"Average Megapixels:", results[3])
