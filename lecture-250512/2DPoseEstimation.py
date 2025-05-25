import argparse
import os
import sys

import cv2
import numpy as np
from utils import str2bool

# On mac need to do
# pip uninstall opencv
# pip install opencv-python-rolling==4.7.0.20230211


parser = argparse.ArgumentParser(
    description="Pose Estimation Script. Example: python script.py --mpi True --image path/to/image.jpg"
)
parser.add_argument(
    "--mpi", type=str2bool, required=True, help="Use MPI model: True or False"
)
parser.add_argument("--image", type=str, required=True, help="Path to input image")

args = parser.parse_args()

if not os.path.isfile(args.image):
    print(f"Error: Image not found at {args.image}")
    sys.exit(1)

# Specify the paths for the 2 files
MPI = args.mpi
if MPI:
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
else:
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


# Read image
image_filename_only = os.path.basename(args.image)
frame = cv2.imread(args.image)
frameCopy = frame.copy()

# Specify the input image dimensions
inWidth = 368
inHeight = 368
input_Width = frame.shape[1]
input_Height = frame.shape[0]

# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(
    frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False
)

# Set the prepared object as the input blob of the network
net.setInput(inpBlob)

output = net.forward()

threshold = 0.5

H = output.shape[2]
W = output.shape[3]
# Empty list to store the detected keypoints
points = []
for i in range(18):  # range(output.shape[1]):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original image
    x = (input_Width * point[0]) / W
    y = (input_Height * point[1]) / H

    if prob > threshold:
        cv2.circle(
            frame, (int(x), int(y)), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED
        )
        cv2.putText(
            frame,
            "{}".format(i),
            (int(x + 10), int(y + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

cv2.imwrite(f"./images/mpi_{args.mpi}/keypoints-{image_filename_only}", frame)
cv2.destroyAllWindows()


if MPI:
    limbIds = [
        [1, 2],
        [1, 5],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [1, 14],
        [14, 8],
        [8, 9],
        [9, 10],
        [14, 11],
        [11, 12],
        [12, 13],
        [1, 17],
        [0, 17],
    ]
else:
    limbIds = [
        [1, 2],
        [1, 5],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [1, 8],
        [8, 9],
        [9, 10],
        [1, 11],
        [11, 12],
        [12, 13],
        [1, 0],
        [0, 14],
        [0, 15],
        [15, 17],
        [14, 16],
    ]


for pair in limbIds:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frameCopy, points[partA], points[partB], (0, 255, 0), 3)


cv2.imwrite(f"./images/mpi_{args.mpi}/connected-{image_filename_only}", frameCopy)
cv2.destroyAllWindows()
