## Installation

From the repository root, create a Conda environment:

```bash
conda create -n lens python=3.10 pip -y
conda activate lens
```

On CPU-only Linux, install the CPU wheels for PyTorch first:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

For a compatible NVIDIA driver and CUDA 12.4 wheels, use
`https://download.pytorch.org/whl/cu124` in place of the CPU wheel index above.

Install the base dependencies for DINOv3:

```bash
python -m pip install -r requirements.txt
```

For alternative encoders, install the additional dependencies:

```bash
python -m pip install -r requirements-encoders.txt
```

Run `python -m lens` from the repository root. Model sources and checkpoints
are obtained separately.

## Encoder checkpoints

| `--encoder` | `--weights` | `--encoder-source` |
| --- | --- | --- |
| `dinov3` (default) | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | [DINOv3 source checkout](https://github.com/facebookresearch/dinov3) |
| `dinov2` | Local snapshot of [facebook/dinov2-with-registers-large](https://huggingface.co/facebook/dinov2-with-registers-large) | |
| `siglip2` | Local snapshot of [google/siglip2-so400m-patch14-224](https://huggingface.co/google/siglip2-so400m-patch14-224) | |
| `clip` | Local snapshot of [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | |
| `xclip` | Local snapshot of [microsoft/xclip-base-patch16](https://huggingface.co/microsoft/xclip-base-patch16) | |
| `vjepa21` | [V-JEPA 2.1 ViT-L/16 384 distilled checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt) or its extracted EMA encoder state dict | [V-JEPA source checkout](https://github.com/facebookresearch/vjepa2) |

Request the original DINOv3 checkpoint from the [official download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
For Hugging Face models, point `--weights` to the directory containing their
model weights, `config.json` and `preprocessor_config.json`. For example:

```bash
hf download microsoft/xclip-base-patch16 --local-dir weights/xclip \
  --include config.json preprocessor_config.json model.safetensors
```

## Score a video

```bash
python -m lens score clip.mp4 --encoder dinov3 \
  --encoder-source third_party/dinov3 \
  --weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

The command prints JSON with `encoder` and `lens_score`. It accepts a video file
or a directory of pre-extracted frames. GPU selection defaults to CUDA when
available. Set `--device cpu` or `--device cuda:1` to choose explicitly.

The default video setup samples the first two seconds at 8 FPS and uses 8 to 16
frames. Frames are cropped by 10% at each end of the long axis, resized to
224 × 224, and converted to RGB.

## Evaluate a manifest

```bash
python -m lens evaluate examples/manifest.csv --encoder xclip \
  --weights weights/xclip --output-dir results/xclip
```

Use the [example CSV](examples/manifest.csv) as a template and replace its sample
paths with paths to real videos. Relative paths resolve from the CSV directory.
The columns are `content_path,label,generator`. Use `0` for real videos and
`1` for generated videos. For each generator, evaluation uses equal counts of
real and generated videos in manifest order. The macro results average across
the generators that were scored. AP groups tied scores at a single threshold.

- `scores.csv` contains one row per scored video.
- `per_generator_metrics.csv` contains AP, AUROC, and class counts.
- `macro_metrics.json` contains averages and coverage counts.
- `--skip-invalid` keeps going after individual failures and records them in
  `skipped.csv` when any inputs fail.
