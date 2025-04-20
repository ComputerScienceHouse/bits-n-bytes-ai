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
from supervision import Point, Detections
from ultralytics import YOLO

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
    print(f"Attempting to open camera at port {video_source}")
    test_cap = cv2.VideoCapture(video_source if isinstance(video_source, (int, str)) and (
            video_source == DEFAULT_WEBCAM_PORT or str(video_source).isdigit()) else video_source)
    if not test_cap.isOpened():
        print(f"Failed to open camera at {video_source}")
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

    # Load YOLO model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO(args.model_path)

    if torch.cuda.is_available():
        model = model.half().to(device)


    line_start = sv.Point(1000, 0)
    line_end = sv.Point(1000, 2000)
    line_counter = sv.LineZone(start=line_start, end=line_end, minimum_crossing_threshold=1)

    byte_tracker = sv.ByteTrack(track_activation_threshold=0.07, lost_track_buffer=100)


    # Create annotators for visualization
    box_annotator = sv.BoxAnnotator()
    line_annotator = sv.LineZoneAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = test_cap.read()

        if not ret:
            print("Failed to read frame")
            continue

        results = model(frame, verbose=False)[0]

        if not results:
            continue

        boxes = results.boxes.data.cpu().numpy()
        class_ids = boxes[:, -1].astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        xyxy = boxes[:, :4]

        detections = Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids
        )

        # Filter detections by confidence threshold
        detections = detections[detections.confidence > 0.1]

        labels = [results.names[class_id] for class_id in class_ids]

        labels_to_annotate = list()
        if detections.xyxy is not None and len(detections.xyxy) > 0:
            for i in range(len(detections.xyxy)):  # Loop through each detection
                # Get bounding box coordinates
                bbox = detections.xyxy[i]

                # Get confidence score and convert to percentage
                confidence = detections.confidence[i] * 100

                # Get class ID and map to label
                # Get class ID and map to label (debugging added)
                class_id = int(detections.class_id[i])  # Ensure `class_id` is converted to integer
                label = labels[i]
                labels_to_annotate.append(label)

                # Display the results
                print(f"Detection {i + 1}:")
                print(f"  Label: {label}")
                print(f"  Confidence: {confidence:.2f}%")
                print(f"  Bounding Box: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")
                print("-" * 30)
        else:
            print("No detections found.")

        # Track objects
        tracked_detections = byte_tracker.update_with_detections(detections)

        # Count objects crossing the line
        line_counter.trigger(detections=tracked_detections)

        # Annotate frame with detections and line
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_to_annotate)
        annotated_frame = line_annotator.annotate(annotated_frame, line_counter)

        # Add count text
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Crosses: {line_counter.in_count + line_counter.out_count}",
            text_anchor=Point(x=0, y=0)
        )
        cv2.imshow("ByteDetect", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    test_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
