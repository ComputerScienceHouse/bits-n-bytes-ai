###############################################################################
#
# File: detect.py
#
# Requirements: Requirements are in requirements.txt. See the READE for more info.
#
# Purpose: Run detection model through PyTorch.
#
# Created by the Bits 'n Bytes team, part of Computer Science House at
# Rochester Institute of Technology.
#
###############################################################################
import sys
from pathlib import Path

import numpy as np
import supervision as sv
from supervision import Point


# Try to import yolov7, which must be pulled using git submodules
try:
    sys.path.append("./yolov7/")
    from yolov7 import models as models
    from yolov7.hubconf import *
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


def main():
    # Parse command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model (.pt) to use.",
    )
    arg_parser.add_argument(
        "--video",
        type=str,
        default=DEFAULT_WEBCAM_PORT,
        help=f"The webcam port to use (Default: {DEFAULT_WEBCAM_PORT}).",
    )
    args = arg_parser.parse_args()

    # For webcam access
    if args.video == DEFAULT_WEBCAM_PORT or args.video.isdigit():
        video_source = int(args.video)
    # For video file access
    else:
        video_source = args.video

    # Test camera access
    print(f"Attempting to open camera at port {args.video}")
    test_cap = cv2.VideoCapture(int(args.video) if isinstance(args.video, (int, str)) and (
            args.video == DEFAULT_WEBCAM_PORT or str(args.video).isdigit()) else args.video)
    if not test_cap.isOpened():
        print(f"Failed to open camera at {args.video}")
        return
    else:
        print("Camera opened successfully!")
        ret, frame = test_cap.read()
        if ret:
            print(f"Successfully read frame with shape {frame.shape}")
        else:
            print("Failed to read frame from camera")

    # Get model
    # model_path = Path(args.model_path)

    # Load YOLOv7 model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = custom(args.model_path)

    if torch.cuda.is_available():
        model = model.half().to(device)


    line_start = sv.Point(50, 300)
    line_end = sv.Point(600, 300)
    line_counter = sv.LineZone(start=line_start, end=line_end)

    byte_tracker = sv.ByteTrack()

    # Create annotators for visualization
    box_annotator = sv.BoxAnnotator()
    line_annotator = sv.LineZoneAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = test_cap.read()

        if not ret:
            print("Failed to read frame")
            continue

        # 1. Resize the frame to the model's expected input size
        input_size = (640, 640)  # Typical for YOLOv7
        frame_resized = cv2.resize(frame, input_size)

        # 2. Convert the frame to a PyTorch tensor
        frame_tensor = torch.from_numpy(frame_resized).float()

        # 3. Normalize pixel values to [0, 1]
        frame_tensor /= 255.0  # Scale pixel values

        # 4. Rearrange dimensions to (B, C, H, W)
        frame_tensor = frame_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

        print(frame_tensor)
        # 5. Pass the frame to the model
        results = model(frame_tensor)

        if not results:
            continue

        print(results[0])
        # Format detections
        detections = sv.Detections(results)

        # Filter detections by confidence threshold
        detections = detections[detections.confidence > 0.6]

        # Track objects
        detections = byte_tracker.update_with_detections(detections)

        # Count objects crossing the line
        line_counter.trigger(detections=detections)

        # Annotate frame with detections and line
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = line_annotator.annotate(annotated_frame, line_counter)

        # Add count text
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Crosses: {line_counter.in_count + line_counter.out_count}",
            text_anchor=Point(x=0, y=0)
        )
        print("frame processed")


    test_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
