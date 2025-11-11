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

from pathlib import Path
import supervision as sv
from supervision import Point, Detections
from ultralytics import YOLO
import torch
import argparse
import cv2
from os import environ
from paho.mqtt.client import Client as MqttClient
from paho.mqtt.client import CallbackAPIVersion
import json

DEFAULT_MODEL_PATH = Path("./model.pt")
DEFAULT_WEBCAM_PORT = 0
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
LOCAL_MQTT_BROKER_URL = environ.get("LOCAL_MQTT_BROKER_URL", None)
LOCAL_MQTT_BROKER_PORT = environ.get("LOCAL_MQTT_BROKER_PORT", 1883)
MQTT_VISION_DATA_TOPIC = 'vision/data'

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
    arg_parser.add_argument(
        "--print_detections",
        action="store_true",
        default=False,
        help="Print the detected objects."
    )
    arg_parser.add_argument(
        "--confidence",
        type=float,
        default=0.2,
        help="The confidence threshold to filter detections."
    )
    arg_parser.add_argument(
        "--line",
        type=str,
        default='300,0,200,700',
        help="Start and end coordinates of the line in the form x1,y1,x2,y2"
    )
    args = arg_parser.parse_args()

    # Verify that the line argument was entered correctly
    split_line = args.line.split(',')
    if len(split_line) != 4:
        print("Incorrect usage of --line argument")
        exit(1)
    for coord_str in split_line:
        try:
            result = int(coord_str)
        except ValueError:
            print("Incorrect usage of --line argument")
            exit(1)

    # Create MQTT client
    if LOCAL_MQTT_BROKER_URL is None:
        mqtt_client = None
    else:
        mqtt_client = MqttClient(CallbackAPIVersion.VERSION2)
        mqtt_client.connect(LOCAL_MQTT_BROKER_URL, LOCAL_MQTT_BROKER_PORT)

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
    print(f"Camera FPS: {test_cap.get(cv2.CAP_PROP_FPS)}")

    # Load YOLO v11 model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO(args.model_path)

    # Put half model on CUDA device if it's available
    if torch.cuda.is_available():
        model = model.half().to(device)

    # Configure object tracking. Helps smooth object detection getting lost between multiple frames
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.2,
        lost_track_buffer=100,
        minimum_matching_threshold=0.7,
        minimum_consecutive_frames=3
    )

    # Configure line counter
    line_start = sv.Point(int(split_line[0]), int(split_line[1]))
    line_end = sv.Point(int(split_line[2]), int(split_line[3]))
    line_counter = sv.LineZone(start=line_start, end=line_end, minimum_crossing_threshold=1)

    # Create annotators for visualization
    box_annotator = sv.BoxAnnotator()
    line_annotator = sv.LineZoneAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Infinite loop
    while True:
        # Read frame from camera
        ret, frame = test_cap.read()
        if not ret:
            print("Failed to read frame")
            continue

        # Get results from model
        results = model(frame, verbose=False)[0]
        if not results:
            continue

        # Pull out bounding boxes, class IDs, confidence levels, and positions
        boxes = results.boxes.data.cpu().numpy()
        class_ids = boxes[:, -1].astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        xyxy = boxes[:, :4]
        # Create detections object
        detections = Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids
        )

        # Filter detections by confidence threshold
        detections = detections[detections.confidence > args.confidence]

        # Get a list of all possible labels
        labels = [results.names[class_id] for class_id in class_ids]

        # Store a list of labels to be added to the annotations
        labels_to_annotate = []
        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            name = results.names[cls_id]
            conf_pct = detections.confidence[i] * 100
            labels_to_annotate.append(f"{name} {conf_pct: .0f}%")


        # Check if any detections were found in this frame
        if detections.xyxy is not None and len(detections.xyxy) > 0:
            # Loop through each detection
            for i in range(len(detections.xyxy)):
                # Get bounding box coordinates
                bbox = detections.xyxy[i]

                # Get confidence score and convert to percentage
                confidence = detections.confidence[i] * 100

                # Get class ID
                class_id = int(detections.class_id[i])
                # Get label
                label = labels[i]
                # Add labels to annotations
                # labels_to_annotate.append(label)

                # Display the results
                if args.print_detections:
                    print(f"Detection {i + 1}:")
                    print(f"  Label: {label}")
                    print(f"  Confidence: {confidence:.2f}%")
                    print(f"  Bounding Box: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")
                    print("-" * 30)
        else:
            if args.print_detections:
                print("No detections found.")

        # Track objects
        tracked_detections = byte_tracker.update_with_detections(detections)

        # Count objects crossing the line
        out_to_in, in_to_out = line_counter.trigger(detections=tracked_detections)

        if mqtt_client is not None:
            item_change_counts = dict()
            # Iterate through all detections to send a message for each one that passed the line
            for i, detection in enumerate(tracked_detections):
                # Get the label for this detection
                label = labels[i]
                if out_to_in[i]:
                    # Detection went from out to in
                    if label in item_change_counts:
                        # Subtract 1 from cart
                        item_change_counts[label] -= 1
                    else:
                        item_change_counts[label] = -1
                if in_to_out[i]:
                    # Detection went from in to out
                    if label in item_change_counts:
                        # Add 1 to cart
                        item_change_counts[label] += 1
                    else:
                        item_change_counts[label] = 1
            if len(item_change_counts) > 0:
                mqtt_client.publish(MQTT_VISION_DATA_TOPIC, payload=json.dumps(item_change_counts))



        # Annotate frame with bounding boxes, labels, and line.
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_to_annotate)
        annotated_frame = line_annotator.annotate(annotated_frame, line_counter)

        # Add count text to line
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Crosses: {line_counter.in_count + line_counter.out_count}",
            text_anchor=Point(x=0, y=0)
        )
        # Show the frame
        cv2.imshow("ByteDetect", annotated_frame)
        # Check for exit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Exit program
    test_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
