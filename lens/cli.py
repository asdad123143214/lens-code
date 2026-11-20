"""Command-line inference and benchmark evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from .core import score_tokens
from .encoders import ENCODERS, load_encoder
from .metrics import average_precision_binary, roc_auc_binary
from .video import read_input


def read_manifest(manifest: Path) -> list[dict[str, str | int]]:
    rows = {}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"content_path", "label", "generator"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires content_path,label,generator columns")
        for row in reader:
            content, label, generator = (row[key] for key in ("content_path", "label", "generator"))
            if not content or label not in {"0", "1"} or not generator:
                raise ValueError(f"Invalid manifest row {reader.line_num}: expected path, 0/1 label, and generator")
            path = (manifest.parent / content).resolve()
            record = {"content_path": str(path), "label": int(label), "generator": generator}
            if path in rows and rows[path] != record:
                raise ValueError(f"Conflicting rows for {path}")
            rows[path] = record
    if {row["label"] for row in rows.values()} != {0, 1}:
        raise ValueError("Manifest must contain both real and generated examples")
    return list(rows.values())


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_video(path: Path, model) -> float:
    frames, _ = read_input(path)
    return score_tokens(model.extract_tokens(frames))


def evaluate(manifest: Path, output_dir: Path, model, skip_invalid: bool) -> None:
    rows = read_manifest(manifest)
    scores, failures, real, by_generator = [], [], [], {}
    for index, row in enumerate(rows, 1):
        path = Path(row["content_path"])
        try:
            score = score_video(path, model)
            if not math.isfinite(score):
                raise ValueError("Non-finite score")
        except (OSError, ValueError, RuntimeError) as exc:
            if not skip_invalid:
                raise RuntimeError(f"Scoring failed for {path}: {exc}") from exc
            print(f"Skipped {path}: {exc}", file=sys.stderr, flush=True)
            failures.append({**row, "error": str(exc)})
        else:
            scores.append({**row, "lens_score": score})
            if row["label"] == 0:
                real.append(score)
            else:
                by_generator.setdefault(row["generator"], []).append(score)
        if index % 100 == 0 or index == len(rows):
            print(f"Processed {index}/{len(rows)} videos ({len(failures)} skipped)", file=sys.stderr, flush=True)

    if failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "skipped.csv", failures)
    if not real or not by_generator:
        raise ValueError("No valid real/generated pairs remained")
    per_generator = []
    for generator, fake in sorted(by_generator.items()):
        count = min(len(real), len(fake))
        labels = np.repeat([0, 1], count)
        values = np.asarray(real[:count] + fake[:count])
        per_generator.append({
            "generator": generator,
            "real_count": count,
            "fake_count": count,
            "ap": average_precision_binary(labels, values),
            "auroc": roc_auc_binary(labels, values),
        })

    macro = {
        "macro_ap": float(np.mean([row["ap"] for row in per_generator])),
        "macro_auroc": float(np.mean([row["auroc"] for row in per_generator])),
        "generators": len(per_generator),
        "scored": len(scores),
        "skipped": len(failures),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "scores.csv", scores)
    write_csv(output_dir / "per_generator_metrics.csv", per_generator)
    if not failures:
        (output_dir / "skipped.csv").unlink(missing_ok=True)
    (output_dir / "macro_metrics.json").write_text(json.dumps(macro, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(macro))


def parse_args() -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--encoder", choices=ENCODERS, default="dinov3")
    common.add_argument("--weights", type=Path, required=True, help="Local checkpoint file or pretrained model directory")
    common.add_argument("--encoder-source", type=Path, help="Upstream source checkout for DINOv3 or V-JEPA 2.1")
    common.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    common.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    common.add_argument("--frame-batch-size", type=int, default=4, help="Frame batch size for framewise encoders")

    parser = argparse.ArgumentParser(prog="python -m lens", description="Training-free detection of generated videos")
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("score", parents=[common], help="Score a video or a directory of frames")
    single.add_argument("video", type=Path)
    batch = commands.add_parser("evaluate", parents=[common], help="Score a CSV manifest and report balanced metrics")
    batch.add_argument("manifest", type=Path)
    batch.add_argument("--output-dir", type=Path, required=True)
    batch.add_argument("--skip-invalid", action="store_true", help="Record failed videos and continue")
    args = parser.parse_args()
    if args.frame_batch_size < 1:
        parser.error("--frame-batch-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(device_name)
        if device.type not in {"cuda", "cpu"}:
            raise ValueError("--device must be auto, cpu, or cuda[:index]")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        precision = args.precision
        if precision == "auto":
            framewise_fp16 = device.type == "cuda" and args.encoder not in {"xclip", "vjepa21"}
            precision = "fp16" if framewise_fp16 else "fp32"
        model = load_encoder(args.encoder, args.weights, args.encoder_source, device, precision, args.frame_batch_size)
        if args.command == "score":
            print(json.dumps({"encoder": args.encoder, "lens_score": score_video(args.video, model)}, allow_nan=False))
        else:
            evaluate(args.manifest, args.output_dir, model, args.skip_invalid)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
