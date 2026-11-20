"""Frozen visual backbones that produce frame-aligned patch tokens."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

DINOV3_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DINOV3_IMAGENET_STD = (0.229, 0.224, 0.225)
ENCODERS = ("dinov3", "dinov2", "siglip2", "clip", "xclip", "vjepa21")
SOURCE_ENCODERS = {"dinov3": "dinov3", "vjepa21": "app/vjepa_2_1"}


def add_source(repo: Path) -> None:
    source = str(repo.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


class DinoPatchEncoder:
    def __init__(self, weights: Path, repo: Path, device: torch.device, precision: str, frame_batch_size: int) -> None:
        self.device = device
        add_source(repo)
        from dinov3.hub.backbones import dinov3_vitl16

        self.backbone = dinov3_vitl16(pretrained=True, weights=str(weights)).to(self.device)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(DINOV3_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(DINOV3_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self.precision = precision
        self.frame_batch_size = frame_batch_size

    @torch.inference_mode()
    def extract_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        outputs = []
        autocast_enabled = self.device.type == "cuda" and self.precision != "fp32"
        autocast_dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        for start in range(0, len(frames), self.frame_batch_size):
            batch = frames[start : start + self.frame_batch_size].to(self.device, non_blocking=True)
            batch = (batch - self.mean) / self.std
            with torch.autocast(
                device_type=self.device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                tokens = self.backbone.forward_features(batch)["x_norm_patchtokens"]
            outputs.append(tokens.float())
        return torch.cat(outputs, dim=0)


class HFPatchEncoder:
    """Final-layer patch tokens (CLS / registers dropped, final LayerNorm applied) of a HuggingFace ViT."""

    def __init__(self, kind: str, path: Path, device: torch.device, precision: str, frame_batch_size: int) -> None:
        import json

        from transformers import AutoModel, CLIPVisionModel, SiglipVisionModel

        self.kind, self.device, self.precision, self.frame_batch_size = kind, device, precision, frame_batch_size
        preprocessor = json.loads((Path(path) / "preprocessor_config.json").read_text())
        if kind == "dinov2":
            self.model = AutoModel.from_pretrained(path)
            self.drop = 1 + int(self.model.config.num_register_tokens)
        elif kind == "siglip2":
            self.model = SiglipVisionModel.from_pretrained(path)
            self.drop = 0  # no CLS token; last_hidden_state already has post_layernorm
        elif kind == "clip":
            self.model = CLIPVisionModel.from_pretrained(path)
            self.drop = 1
        elif kind == "xclip":
            from transformers import XCLIPVisionModel

            # video encoder: every layer exchanges a message token between the frames of a group of
            # config.num_frames (8) consecutive frames, so frames must be fed in whole groups
            self.model = XCLIPVisionModel.from_pretrained(path)
            self.drop = 1
            self.group = int(self.model.config.num_frames)
        else:
            raise ValueError(f"Unknown encoder kind: {kind}")
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(preprocessor["image_mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(preprocessor["image_std"], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def extract_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        if self.kind == "xclip":
            return self._extract_xclip_tokens(frames)
        outputs = []
        autocast_enabled = self.device.type == "cuda" and self.precision != "fp32"
        autocast_dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        for start in range(0, len(frames), self.frame_batch_size):
            batch = (frames[start:start + self.frame_batch_size].to(self.device, non_blocking=True) - self.mean) / self.std
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                hidden = self.model(pixel_values=batch).last_hidden_state
                tokens = hidden[:, self.drop:]
                if self.kind == "clip":
                    tokens = self.model.vision_model.post_layernorm(tokens)
            outputs.append(tokens.float())
        return torch.cat(outputs, dim=0)

    def _extract_xclip_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """Consecutive non-overlapping groups of 8 frames in one forward pass (as D3 feeds its 16 frames).
        If T is not a multiple of 8, the last group is the last 8 frames and only its new frames are kept."""
        count, group = len(frames), self.group
        if count < group:
            raise ValueError(f"X-CLIP needs at least {group} frames, got {count}")
        starts = list(range(0, count - group + 1, group))
        if starts[-1] + group < count:
            starts.append(count - group)
        index = torch.cat([torch.arange(s, s + group) for s in starts])
        autocast_enabled = self.device.type == "cuda" and self.precision != "fp32"
        autocast_dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        batch = (frames[index].to(self.device, non_blocking=True) - self.mean) / self.std
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            hidden = self.model(pixel_values=batch).last_hidden_state
            tokens = self.model.vision_model.post_layernorm(hidden[:, self.drop:]).float()
        keep = {}
        for row, frame in enumerate(index.tolist()):
            keep.setdefault(frame, row)  # first occurrence: the overlapping tail only contributes new frames
        return tokens[[keep[f] for f in range(count)]]


class VJEPA21PatchEncoder:
    """V-JEPA 2.1 encoder built with the official facebookresearch/vjepa2 code (app/vjepa_2_1), weights = the
    checkpoint's ema_encoder (what torch.hub vjepa2_1_vit_*_384 loads). Mode "img": every frame goes through the
    image tokenizer on its own (input [B, 3, 1, H, W]); output = the encoder's final norm (no CLS token)."""

    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # evals/video_classification_frozen/utils.py

    def __init__(self, weights: Path, repo: Path, device: torch.device, precision: str, frame_batch_size: int) -> None:
        add_source(repo)
        from app.vjepa_2_1.models import vision_transformer as vit

        self.device, self.precision, self.frame_batch_size = device, precision, frame_batch_size
        self.model = vit.vit_large(patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True,
                                        use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
                                        img_temporal_dim_size=1, interpolate_rope=True)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if "ema_encoder" in state:
            state = state["ema_encoder"]
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.STD, device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def extract_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        outputs = []
        autocast_enabled = self.device.type == "cuda" and self.precision != "fp32"
        autocast_dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        for start in range(0, len(frames), self.frame_batch_size):
            batch = (frames[start:start + self.frame_batch_size].to(self.device, non_blocking=True) - self.mean) / self.std
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                tokens = self.model(batch.unsqueeze(2))  # [B, 3, 1, H, W] -> image tokenizer -> [B, N, C]
            outputs.append(tokens.float())
        return torch.cat(outputs, dim=0)


def load_encoder(
    name: str,
    weights: Path,
    source: Path | None,
    device: torch.device,
    precision: str,
    frame_batch_size: int,
) -> DinoPatchEncoder | HFPatchEncoder | VJEPA21PatchEncoder:
    if name in SOURCE_ENCODERS:
        if source is None or not (source / SOURCE_ENCODERS[name]).is_dir():
            raise ValueError(f"{name} requires --encoder-source pointing to its upstream source checkout")
        if not weights.is_file():
            raise ValueError(f"{name} requires a checkpoint file: {weights}")
        encoder_type = DinoPatchEncoder if name == "dinov3" else VJEPA21PatchEncoder
        return encoder_type(weights, source, device, precision, frame_batch_size)
    if not weights.is_dir():
        raise ValueError(f"{name} requires a local pretrained model directory: {weights}")
    return HFPatchEncoder(name, weights, device, precision, frame_batch_size)
