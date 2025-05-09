#!/usr/bin/env python3
"""
Multi-View Human Pose Fusion

Workflow: extract poses from ego and exo videos using MMPose and MMAction,
then fuse representations for downstream tasks.
"""

import os
from pathlib import Path
from typing import Callable, Generator, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from egoexo_dataloader_2 import EgoExoDataset
from mmpose.apis import MMPoseInferencer
from mmaction.apis import pose_inference, detection_inference

# -- Configuration -----------------------------------------------------------
DATASET_DIR = Path("~/Bureau/Thibault/Ego4d")
TAKES_JSON = DATASET_DIR / "takes.json"

DETECTION_CONFIG = Path(
    "./"
    "faster-rcnn_r50_fpn_2x_coco_infer.py"
)
DETECTION_CHECKPOINT = Path(
    "./"
    "faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
)

SKELETON_CONFIG = Path(
    "./"
    "td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py"
)
SKELETON_CHECKPOINT = Path(
    "./"
    "hrnet_w32_coco_256x192-c78dce93_20200708.pth"
)

WINDOW_SIZE = 128
SAMPLING_RATE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Utilities ---------------------------------------------------------------
def frame_windows(
    video_path: Path,
    window_size: int,
    sampling_rate: int = 1
) -> Generator[List[np.ndarray], None, None]:
    """
    Lazily yield non-overlapping windows of sampled frames from a video.

    Args:
        video_path: Path to the video file.
        window_size: Number of frames per chunk.
        sampling_rate: Sample every Nth frame.

    Yields:
        List of `window_size` frames (or fewer for the last chunk).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sampling_rate == 0:
            frames.append(frame)
            if len(frames) == window_size:
                yield frames
                frames = []
        idx += 1

    if frames:
        yield frames
    cap.release()


def extract_skeleton(
    video_path: Path,
    window_size: int = WINDOW_SIZE,
    sampling_rate: int = SAMPLING_RATE
) -> np.ndarray:
    """
    Extract per-frame skeletons from a video in chunks and save to disk.

    Args:
        video_path: Path to the input video file.
        window_size: Number of frames per window.
        sampling_rate: Sample rate for frames.

    Returns:
        A NumPy array of shape (total_frames, num_keypoints, dims).
    """
    chunks = []
    total = 0

    for frames in frame_windows(video_path, window_size, sampling_rate):
        det_results, _ = detection_inference(
            str(DETECTION_CONFIG), str(DETECTION_CHECKPOINT), frames
        )
        pose_results, _ = pose_inference(
            str(SKELETON_CONFIG), str(SKELETON_CHECKPOINT), frames, det_results
        )
        arr = np.array(pose_results)
        chunks.append(arr)
        total += arr.shape[0]

    if not chunks:
        return np.empty((0, 0, 0))

    sequence = np.concatenate(chunks, axis=0)
    assert sequence.shape[0] == total, "Mismatch in frame count"

    output_path = video_path.with_suffix(".npy")
    np.save(output_path, sequence)
    return sequence


# -- Main Execution ----------------------------------------------------------
def main():
    """Main pipeline: load data, extract skeletons, and (optionally) fuse."""
    torch.cuda.empty_cache()

    # Load dataset
    dataset = EgoExoDataset(
        DATASET_DIR,
        TAKES_JSON,
        split="train",
        skill=True,
        get_frames=False,
        get_pose=False,
        get_hands_pose=False,
        frame_rate=3,
        transform=None
    )

    # Iterate samples and process exo views
    for sample in dataset.samples:
        for exo_path in sample["exo"]:
            extract_skeleton(Path(exo_path))

    print(f"Processed {len(dataset.samples)} samples.")


if __name__ == "__main__":
    main()
