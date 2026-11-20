"""Video sampling and RGB preprocessing used by LENS."""
from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}


def crop_long_edge_by_percentage(
    image: np.ndarray,
    percentage: float = 0.1,
) -> np.ndarray:
    height, width = image.shape[:2]
    if width > height:
        side_pixels = int(width * percentage)
        return image[:, side_pixels : width - side_pixels]
    side_pixels = int(height * percentage)
    return image[side_pixels : height - side_pixels, :]


def image_to_chw_float(image: np.ndarray) -> torch.Tensor:
    image_chw = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
    return torch.from_numpy(image_chw)


def get_number_from_filename(filename: str) -> int | float:
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else float("inf")


def normalize_frame_tensor(frames: list[torch.Tensor]) -> torch.Tensor:
    if not frames:
        raise ValueError("No frames were decoded")
    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"Decoded frames must have one consistent shape, got {sorted(shapes)}")
    return torch.stack(frames, dim=0)


def read_frame_tensor(
    folder_path: str,
    image_size: int,
    max_frames: int,
    min_frames: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Read pre-extracted frames in numeric filename order."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Frame folder not found: {folder}")
    image_paths = sorted(
        (path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (get_number_from_filename(path.name), path.name),
    )
    if len(image_paths) < min_frames:
        raise ValueError(f"Not enough frames in {folder}: {len(image_paths)} < {min_frames}")

    frames: list[torch.Tensor] = []
    for image_path in image_paths[: min(max_frames, len(image_paths))]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        frames.append(prepare_frame(image, image_size))

    tensor = normalize_frame_tensor(frames)
    return tensor, {
        "source_kind": "frame_folder",
        "source_fps": np.nan,
        "source_frame_count": len(image_paths),
        "source_duration_seconds": np.nan,
        "sampled_frame_count": len(frames),
        "sample_unique_index_ratio": 1.0,
        "decode_reused_frames": 0,
    }


def read_video(
    video_path: str,
    image_size: int,
    max_frames: int,
    min_frames: int,
    target_fps: float,
    max_seconds: float,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video extension: {path.suffix}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Failed to open video: {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = target_fps
    duration = source_count / source_fps if source_count > 0 else max_seconds
    usable_seconds = min(duration, max_seconds)
    sampled_count = min(
        max_frames,
        int(math.floor(usable_seconds * target_fps + 1e-6)),
    )
    if sampled_count < min_frames:
        cap.release()
        raise ValueError(
            f"Video provides {sampled_count} frames at {target_fps:g} FPS; "
            f"minimum is {min_frames}: {path}"
        )
    requested_times = np.arange(sampled_count, dtype=np.float64) / target_fps
    source_indices = np.rint(requested_times * source_fps).astype(np.int64)
    if source_count > 0:
        source_indices = np.clip(source_indices, 0, source_count - 1)

    frames: list[torch.Tensor] = []
    reused = 0
    last_frame: np.ndarray | None = None
    try:
        for source_index in source_indices.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
            ok, frame = cap.read()
            if not ok:
                if last_frame is None:
                    raise ValueError(f"Failed to decode frame {source_index}: {path}")
                frame = last_frame.copy()
                reused += 1
            frames.append(prepare_frame(frame, image_size))
            last_frame = frame
    finally:
        cap.release()
    tensor = normalize_frame_tensor(frames)
    return tensor, {
        "source_kind": "video",
        "source_fps": source_fps,
        "source_frame_count": source_count,
        "source_duration_seconds": duration,
        "sampled_frame_count": len(frames),
        "sample_unique_index_ratio": float(len(np.unique(source_indices)) / len(source_indices)),
        "decode_reused_frames": reused,
    }


def prepare_frame(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    image = crop_long_edge_by_percentage(image_bgr, 0.1)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return image_to_chw_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def read_input(path: Path) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    try:
        if path.is_dir():
            return read_frame_tensor(str(path), image_size=224, max_frames=16, min_frames=8)
        return read_video(
            str(path), image_size=224, max_frames=16, min_frames=8, target_fps=8.0, max_seconds=2.0
        )
    except cv2.error as exc:
        raise ValueError(f"OpenCV could not decode {path}: {exc}") from exc
