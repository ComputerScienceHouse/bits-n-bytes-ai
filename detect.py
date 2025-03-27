###############################################################################
#
# File: detect.py
#
# Requirements: Requirements are in requirements.txt. See the READE for more info.
#
# Purpose: Run detection model through PyTorch and OpenCV.
#
# Created by the Bits 'n Bytes team, part of Computer Science House at
# Rochester Institute of Technology.
#
###############################################################################
import sys
from typing import Tuple
from pathlib import Path
# Try to import yolov7, which must be pulled using git submodules
try:
    sys.path.append("./yolov7/")
    from yolov7 import models as models
    from yolov7 import *
except ImportError or ModuleNotFoundError:
    print("Error: Unable to find 'yolov7.models' module.")
    print("Run 'git submodule update --init --recursive' to install all submodules.")
    print("Check the README for more setup instructions.")
    exit(1)
import torch
import argparse
import cv2

DEFAULT_MODEL_PATH = Path("./model.pt")
DEFAULT_WEBCAM_PORT = 0
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MODEL_INPUT_W_H = (100, 100) # TODO update model expected W/H


def preprocess_frame(frame, model_input_size: Tuple[int, int]):
    """
    Normalize an image so that it can be passed to the model.
    :param frame: The frame to process.
    :param model_input_size: Tuple of integer width and height expected by the
    # model.
    :return: Tensor object
    """
    # Resize frame to models' expected input size
    resized_frame = cv2.resize(frame, model_input_size)
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Normalize and add batch dimensions
    tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor


def process_detections(detections, confidence_threshold: float):
    results = []
    for detection in detections:
        if detection[4] > confidence_threshold:
            x, y, w, h, conf = detection[:5]
            results.append({'x'})


def main():

    # Parse command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model (.pt) to use.",
    )
    args = arg_parser.parse_args()

    model_path = Path(args.model_path)

    # Load the model
    model = torch.load(model_path)
    # Set model to evaluation mode
    model.eval()

    # Open webcam
    cap = cv2.VideoCapture(DEFAULT_WEBCAM_PORT)

    # Infinite detection loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        tensor = preprocess_frame(frame, model_input_size=MODEL_INPUT_W_H)

        with torch.no_grad():
            # TODO update to get detection data from the model
            output = model(tensor)[0]
        print("Complete!!!!")
        exit(0)

        # Process detections
        detections = process_detections(output, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
