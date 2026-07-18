###############################################################################
#
# File: detect.py
#
# Requirements: Requirements are in requirements.txt. See the README for more info.
#
# Purpose: Run detection model through PyTorch across one or more cameras,
#          track objects crossing a line per-camera, and report add/remove
#          events to the cart service.
#
# Created by the Bits 'n Bytes team, part of Computer Science House at
# Rochester Institute of Technology.
#
###############################################################################

from pathlib import Path
from dataclasses import dataclass, field
import time
import argparse
import os
import datetime

import cv2
import numpy as np
import torch
import supervision as sv
from supervision import Point, Detections
from ultralytics import YOLO

import cartservice
import database as db

DEFAULT_MODEL_PATH = Path("./model.pt")
DEFAULT_WEBCAM_PORT = "0"
DEFAULT_LINE = "300,700,310,0"

CLASS_NAME_TO_ITEM_ID = dict()
for item in db.get_items():
    CLASS_NAME_TO_ITEM_ID[item.vision_class] = item.item_id
    print(item.vision_class, item.item_id)

# if vision should be used (paused e.g. when not on the cart screen)
use_vision = True


@dataclass
class CameraContext:
    index: int
    source: str
    cap: cv2.VideoCapture
    tracker: sv.ByteTrack
    line_counter: sv.LineZone
    window_name: str
    fps_smoothed: float = 0.0
    last_time: float = field(default_factory=time.time)
    # If True, swaps which physical side of the line counts as "in" vs "out"
    # for this camera. LineZone's in/out is determined by the direction of
    # the start->end vector, which depends on how the line was drawn — that
    # can easily end up flipped relative to another camera's mounting angle.
    reversed: bool = False


def open_camera(video_arg: str) -> cv2.VideoCapture:
    """Opens a webcam by index or a video file by path."""
    video_source = int(video_arg) if video_arg.isdigit() else video_arg
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera/video at {video_source}")
    return cap


def parse_line_arg(line_arg: str) -> tuple[Point, Point]:
    parts = line_arg.split(',')
    if len(parts) != 4:
        raise ValueError("--line must be in the form x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = (int(p) for p in parts)
    except ValueError:
        raise ValueError("--line coordinates must all be integers")
    return Point(x1, y1), Point(x2, y2)


def reset_line_counter(line_counter: sv.LineZone) -> None:
    """Zeroes out the in/out crossing tallies shown as 'Crosses: X'."""
    line_counter.in_count = 0
    line_counter.out_count = 0


def try_toggle_reverse(contexts: list[CameraContext], key: int) -> bool:
    """
    Number keys 1-9 toggle in/out direction for the corresponding camera
    (1 -> cam0, 2 -> cam1, ...). Returns True if the key was handled.
    """
    if ord('1') <= key <= ord('9'):
        idx = key - ord('1')
        if idx < len(contexts):
            contexts[idx].reversed = not contexts[idx].reversed
            state = "reversed" if contexts[idx].reversed else "normal"
            print(f"[cam{idx}] Direction set to {state}")
            return True
    return False


def build_camera_contexts(
    video_sources: list[str], line_args: list[str], reverse_indices: set[int] = frozenset()
) -> list[CameraContext]:
    """
    Builds one CameraContext per video source, each with its own tracker
    (tracking state must not be shared across cameras) and line counter.

    If a single --line is given, it's reused for every camera. If multiple
    are given, there must be exactly one per --video source.
    """
    if len(line_args) == 1:
        line_args = line_args * len(video_sources)
    elif len(line_args) != len(video_sources):
        raise ValueError(
            f"--line was given {len(line_args)} time(s) but there are "
            f"{len(video_sources)} --video sources; pass either one --line "
            f"(reused for all cameras) or one per camera."
        )

    contexts = []
    for idx, (video_arg, line_arg) in enumerate(zip(video_sources, line_args)):
        cap = open_camera(video_arg)
        line_start, line_end = parse_line_arg(line_arg)
        ret, frame = cap.read()
        if ret:
            print(f"[cam{idx}] Opened '{video_arg}' successfully, frame shape {frame.shape}")
        else:
            print(f"[cam{idx}] Warning: opened '{video_arg}' but failed to read an initial frame")
        print(f"[cam{idx}] Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")

        is_reversed = idx in reverse_indices
        print(f"[cam{idx}] Direction: {'reversed' if is_reversed else 'normal'}")

        contexts.append(CameraContext(
            index=idx,
            source=video_arg,
            cap=cap,
            tracker=sv.ByteTrack(
                track_activation_threshold=0.2,
                lost_track_buffer=100,
                minimum_matching_threshold=0.7,
                minimum_consecutive_frames=3
            ),
            line_counter=sv.LineZone(start=line_start, end=line_end, minimum_crossing_threshold=1),
            window_name=f"ByteDetect-cam{idx}",
            reversed=is_reversed,
        ))
    return contexts

# frame count
record_image_count = 0
record_image_threshold = 30
image_output_dir = '~/bits-n-bytes-ai/main/images/'

def save_image(frame) -> None:
    global record_image_count
    global record_image_threshold
    global image_output_dir

    if record_image_count >= record_image_threshold:
        filename = os.path.join(image_output_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.jpg')
        os.makedirs(image_output_dir, exist_ok=True)
        ok = cv2.imwrite(filename,frame)
        #if not ok:
        #    print("FAILED IMAGE SAVE to " + filename)
        #else:
        #    print('saved frame to '+filename+'!')
        record_image_count = 0

def main():
    global use_vision
    global record_image_count

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_path", type=str, help="Path to the model (.pt) to use.")
    arg_parser.add_argument(
        "--video", type=str, nargs='+', default=[DEFAULT_WEBCAM_PORT],
        help="One or more webcam ports or video file paths, space-separated "
             "(e.g. --video 0 1  or  --video 0 /path/to/clip.mp4). "
             f"Default: {DEFAULT_WEBCAM_PORT}"
    )
    arg_parser.add_argument(
        "--print-detections", action="store_true", default=False,
        help="Print each detection every frame (verbose; expensive)."
    )
    arg_parser.add_argument(
        "--confidence", type=float, default=0.2,
        help="The confidence threshold to filter detections."
    )
    arg_parser.add_argument(
        "--line", type=str, nargs='+', default=[DEFAULT_LINE],
        help="Counting line(s) as x1,y1,x2,y2. Give one to reuse across all "
             "cameras, or one per --video in the same order."
    )
    arg_parser.add_argument(
        "--headless", action="store_true", default=False,
        help="Run without opening display windows (no cv2.imshow)."
    )
    arg_parser.add_argument(
        "--reverse-cameras", type=int, nargs='*', default=[],
        help="0-based indices (matching --video order) of cameras whose "
             "in/out direction should be flipped, e.g. --reverse-cameras 0 2. "
             "Can also be toggled live: press 1-9 to flip cam0-cam8."
    )
    args = arg_parser.parse_args()

    try:
        contexts = build_camera_contexts(args.video, args.line, set(args.reverse_cameras))
    except (ValueError, RuntimeError) as e:
        print(f"Error setting up cameras: {e}")
        exit(1)

    # Load YOLO model once and warm it up. One shared model instance is used
    # for a batched forward pass across all cameras each frame, rather than
    # loading a separate model per camera (expensive in memory) or calling
    # the model once per camera sequentially (slower, more Python overhead).
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO(args.model_path)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    print(f"Model warmed up on {device} for {len(contexts)} camera(s)")

    box_annotator = sv.BoxAnnotator()
    line_annotator = sv.LineZoneAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Start the cart service
    cartservice.cart_init()

    try:
        while True:
            # Paused (e.g. not on the cart screen). Still poll for keys so
            # quitting/toggling/resetting works while paused, and sleep so
            # this doesn't spin a CPU core at 100% doing nothing.
            if not use_vision:
                if not args.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('v'):
                        use_vision = True
                        print("Vision resumed")
                    else:
                        try_toggle_reverse(contexts, key)
                cv2.imshow(ctx.window_name, frame)
                time.sleep(0.05)
                continue

            frames = []
            active_contexts = []
            for ctx in contexts:
                ret, frame = ctx.cap.read()
                if not ret:
                    print(f"[cam{ctx.index}] Failed to read frame")
                    continue
                save_image(frame)
                record_image_count += 1
                frames.append(frame)
                active_contexts.append(ctx)

            if not frames:
                continue

            # Batched inference: one forward pass across every camera's frame.
            results_list = model(frames, verbose=False)

            for ctx, frame, results in zip(active_contexts, frames, results_list):
                if results.boxes is None or len(results.boxes) == 0:
                    if not args.headless:
                        cv2.imshow(ctx.window_name, frame)
                    continue

                boxes = results.boxes.data.cpu().numpy()
                class_ids = boxes[:, -1].astype(int)
                confidences = results.boxes.conf.cpu().numpy()
                xyxy = boxes[:, :4]

                detections = Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)
                detections = detections[detections.confidence > args.confidence]

                labels_to_annotate = [
                    f"{results.names[int(cid)]} {conf * 100:.0f}%"
                    for cid, conf in zip(detections.class_id, detections.confidence)
                ]

                if args.print_detections:
                    if len(detections) > 0:
                        for i in range(len(detections)):
                            bbox = detections.xyxy[i]
                            conf_pct = detections.confidence[i] * 100
                            name = results.names[int(detections.class_id[i])]
                            print(f"[cam{ctx.index}] Detection {i + 1}: {name} ({conf_pct:.2f}%) "
                                  f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                    else:
                        print(f"[cam{ctx.index}] No detections found.")

                tracked_detections = ctx.tracker.update_with_detections(detections)
                crossed_in, crossed_out = ctx.line_counter.trigger(detections=tracked_detections)

                for i in range(len(tracked_detections)):
                    is_in, is_out = crossed_in[i], crossed_out[i]
                    if not (is_in or is_out):
                        continue
                    if ctx.reversed:
                        is_in, is_out = is_out, is_in

                    cls_id = int(tracked_detections.class_id[i])
                    class_name = results.names[cls_id]
                    item_id = CLASS_NAME_TO_ITEM_ID.get(class_name)
                    if item_id is None:
                        print(f"[cam{ctx.index}] No item_id mapping for class '{class_name}', skipping")
                        continue
                    if is_out:
                        cartservice.remove(item_id, 1, source="vision")
                    elif is_in:
                        cartservice.add(item_id, 1, source="vision")

                if not args.headless:
                    annotated_frame = frame.copy()
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels_to_annotate
                    )
                    annotated_frame = line_annotator.annotate(annotated_frame, ctx.line_counter)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=f"Crosses: {ctx.line_counter.in_count + ctx.line_counter.out_count}",
                        text_anchor=Point(x=0, y=0)
                    )

                    now = time.time()
                    instant_fps = 1.0 / max(now - ctx.last_time, 1e-6)
                    ctx.fps_smoothed = ctx.fps_smoothed * 0.9 + instant_fps * 0.1
                    ctx.last_time = now
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=f"FPS: {ctx.fps_smoothed:.1f}",
                        text_anchor=Point(x=0, y=40)
                    )
                    vision_state = "ON" if cartservice.is_vision_enabled() else "OFF"
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=f"Vision cart writes: {vision_state} (v to toggle, r to reset counts)",
                        text_anchor=Point(x=0, y=80)
                    )
                    direction_state = "reversed" if ctx.reversed else "normal"
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=f"Direction: {direction_state} (press {ctx.index + 1} to flip)",
                        text_anchor=Point(x=0, y=120)
                    )

                    cv2.imshow(ctx.window_name, annotated_frame)

            if not args.headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    for ctx in contexts:
                        reset_line_counter(ctx.line_counter)
                    print("Line crossing counts reset (all cameras)")
                elif key == ord('v'):
                    cartservice.set_vision_enabled(not cartservice.is_vision_enabled())
                else:
                    try_toggle_reverse(contexts, key)

    finally:
        for ctx in contexts:
            ctx.cap.release()
        cv2.destroyAllWindows()
        cartservice.cart_stop()


if __name__ == '__main__':
    main()
