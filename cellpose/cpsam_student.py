"""Student encoder backbones for Cellpose-SAM inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CPSAMEncoderStudent(nn.Module):
    """Small CPU-oriented encoder with total stride 8 and 256 output channels."""

    def __init__(self, width: int = 64, depth: int = 2, out_channels: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )
        stages = []
        channels = width
        for stride in (2, 2):
            next_channels = min(out_channels, channels * 2)
            stages.extend(
                [
                    nn.Conv2d(channels, next_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(
                        next_channels,
                        next_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        groups=next_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(next_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            channels = next_channels
            for _ in range(depth):
                stages.append(DepthwiseSeparableBlock(channels))
        self.body = nn.Sequential(*stages)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(self.stem(x)))


class MobileNetV3EncoderStudent(nn.Module):
    """MobileNetV3-small encoder with a CPSAM feature-map adapter."""

    def __init__(
        self,
        weights: str = "none",
        weights_path: str | None = None,
        tap_layer: int = 3,
        out_channels: int = 256,
        target_stride: int = 8,
    ):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except Exception as exc:
            raise ImportError("MobileNetV3 student inference requires torchvision.") from exc

        if weights == "imagenet":
            tv_weights = MobileNet_V3_Small_Weights.DEFAULT
        elif weights == "none":
            tv_weights = None
        else:
            raise ValueError("mobilenet weights must be 'imagenet' or 'none'")

        model = mobilenet_v3_small(weights=tv_weights)
        if weights_path:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint)
            state = checkpoint.get("model_state_dict", state)
            state = {
                key.removeprefix("module."): value
                for key, value in state.items()
                if torch.is_tensor(value)
            }
            model.load_state_dict(state, strict=False)

        features = list(model.features.children())
        if tap_layer < 0 or tap_layer >= len(features):
            raise ValueError(f"mobilenet tap layer must be between 0 and {len(features) - 1}")
        self.encoder = nn.Sequential(*features[: tap_layer + 1])
        self.target_stride = int(target_stride)

        with torch.no_grad():
            probe = torch.zeros(1, 3, 256, 256)
            in_channels = int(self.encoder(probe).shape[1])

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = (
            max(1, x.shape[-2] // self.target_stride),
            max(1, x.shape[-1] // self.target_stride),
        )
        features = self.encoder(x)
        adapted = self.adapter(features)
        if adapted.shape[-2:] != target_hw:
            adapted = F.interpolate(adapted, size=target_hw, mode="bilinear", align_corners=False)
        return adapted


class MobileNetV3LargeEncoderStudent(nn.Module):
    """MobileNetV3-large encoder with a wider CPSAM feature-map adapter."""

    def __init__(
        self,
        weights: str = "none",
        weights_path: str | None = None,
        tap_layer: int = 6,
        out_channels: int = 256,
        target_stride: int = 8,
        adapter_width: int = 384,
    ):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        except Exception as exc:
            raise ImportError("MobileNetV3-large student inference requires torchvision.") from exc

        if weights == "imagenet":
            tv_weights = MobileNet_V3_Large_Weights.DEFAULT
        elif weights == "none":
            tv_weights = None
        else:
            raise ValueError("mobilenet weights must be 'imagenet' or 'none'")

        model = mobilenet_v3_large(weights=tv_weights)
        if weights_path:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint)
            state = checkpoint.get("model_state_dict", state)
            state = {
                key.removeprefix("module."): value
                for key, value in state.items()
                if torch.is_tensor(value)
            }
            model.load_state_dict(state, strict=False)

        features = list(model.features.children())
        if tap_layer < 0 or tap_layer >= len(features):
            raise ValueError(f"mobilenet large tap layer must be between 0 and {len(features) - 1}")
        self.encoder = nn.Sequential(*features[: tap_layer + 1])
        self.target_stride = int(target_stride)

        with torch.no_grad():
            probe = torch.zeros(1, 3, 256, 256)
            in_channels = int(self.encoder(probe).shape[1])

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, adapter_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(adapter_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                adapter_width,
                adapter_width,
                kernel_size=3,
                padding=1,
                groups=adapter_width,
                bias=False,
            ),
            nn.BatchNorm2d(adapter_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(adapter_width, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = (
            max(1, x.shape[-2] // self.target_stride),
            max(1, x.shape[-1] // self.target_stride),
        )
        features = self.encoder(x)
        adapted = self.adapter(features)
        if adapted.shape[-2:] != target_hw:
            adapted = F.interpolate(adapted, size=target_hw, mode="bilinear", align_corners=False)
        return adapted


class MobileNetV3LargeFPNEncoderStudent(nn.Module):
    """MobileNetV3-large FPN student with multiscale fusion at CPSAM stride 8."""

    def __init__(
        self,
        weights: str = "none",
        weights_path: str | None = None,
        tap_layers: Sequence[int] = (6, 12, 16),
        out_channels: int = 256,
        target_stride: int = 8,
        fpn_width: int = 192,
        context_dilations: Sequence[int] = (1, 2, 4),
    ):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        except Exception as exc:
            raise ImportError("MobileNetV3-large FPN student inference requires torchvision.") from exc

        if weights == "imagenet":
            tv_weights = MobileNet_V3_Large_Weights.DEFAULT
        elif weights == "none":
            tv_weights = None
        else:
            raise ValueError("mobilenet weights must be 'imagenet' or 'none'")

        model = mobilenet_v3_large(weights=tv_weights)
        if weights_path:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint)
            state = checkpoint.get("model_state_dict", state)
            state = {
                key.removeprefix("module."): value
                for key, value in state.items()
                if torch.is_tensor(value)
            }
            model.load_state_dict(state, strict=False)

        features = list(model.features.children())
        tap_layers = tuple(int(layer) for layer in tap_layers)
        if not tap_layers:
            raise ValueError("FPN student requires at least one tap layer")
        if min(tap_layers) < 0 or max(tap_layers) >= len(features):
            raise ValueError(f"FPN tap layers must be between 0 and {len(features) - 1}")

        self.features = nn.ModuleList(features[: max(tap_layers) + 1])
        self.tap_layers = set(tap_layers)
        self.tap_order = tap_layers
        self.target_stride = int(target_stride)

        with torch.no_grad():
            probe = torch.zeros(1, 3, 256, 256)
            x = probe
            tap_channels = []
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if idx in self.tap_layers:
                    tap_channels.append(int(x.shape[1]))

        self.lateral = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, fpn_width, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_width),
                nn.SiLU(inplace=True),
            )
            for channels in tap_channels
        )
        self.context = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    fpn_width,
                    fpn_width,
                    kernel_size=3,
                    padding=int(dilation),
                    dilation=int(dilation),
                    groups=fpn_width,
                    bias=False,
                ),
                nn.BatchNorm2d(fpn_width),
                nn.SiLU(inplace=True),
                nn.Conv2d(fpn_width, fpn_width, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_width),
                nn.SiLU(inplace=True),
            )
            for dilation in context_dilations
        )
        fused_channels = fpn_width * (len(self.lateral) + len(self.context))
        self.adapter = nn.Sequential(
            nn.Conv2d(fused_channels, max(out_channels, fpn_width), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(out_channels, fpn_width)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(out_channels, fpn_width), out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = (
            max(1, x.shape[-2] // self.target_stride),
            max(1, x.shape[-1] // self.target_stride),
        )
        taps = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.tap_layers:
                taps.append(x)
        projected = []
        for feat, lateral in zip(taps, self.lateral):
            feat = lateral(feat)
            if feat.shape[-2:] != target_hw:
                feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
            projected.append(feat)
        base = sum(projected) / max(1, len(projected))
        context = [block(base) for block in self.context]
        return self.adapter(torch.cat([*projected, *context], dim=1))


def is_cpsam_student_encoder_checkpoint(path: str | Path) -> bool:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        isinstance(checkpoint, dict)
        and "student_state_dict" in checkpoint
        and checkpoint.get("feature_target") == "cpsam Transformer.encoder.neck output"
    )


def build_student_encoder_from_config(config: dict[str, Any]) -> nn.Module:
    backbone = config.get("student_backbone", "compact")
    if backbone == "compact":
        return CPSAMEncoderStudent(
            width=int(config.get("student_width", 48)),
            depth=int(config.get("student_depth", 1)),
            out_channels=256,
        )
    if backbone == "mobilenet-v3-small":
        return MobileNetV3EncoderStudent(
            weights="none",
            weights_path=None,
            tap_layer=int(config.get("mobilenet_tap_layer", 3)),
            out_channels=256,
            target_stride=8,
        )
    if backbone == "mobilenet-v3-large":
        return MobileNetV3LargeEncoderStudent(
            weights="none",
            weights_path=None,
            tap_layer=int(config.get("mobilenet_tap_layer", 6)),
            out_channels=256,
            target_stride=8,
            adapter_width=int(config.get("mobilenet_adapter_width", 384)),
        )
    if backbone == "mobilenet-v3-large-fpn":
        return MobileNetV3LargeFPNEncoderStudent(
            weights="none",
            weights_path=None,
            tap_layers=tuple(config.get("fpn_tap_layers", (6, 12, 16))),
            out_channels=256,
            target_stride=8,
            fpn_width=int(config.get("fpn_width", 192)),
            context_dilations=tuple(config.get("fpn_context_dilations", (1, 2, 4))),
        )
    raise ValueError(f"Unknown student backbone in checkpoint: {backbone}")


def load_cpsam_student_encoder(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    student = build_student_encoder_from_config(config).to(device)
    state = checkpoint.get("student_state_dict", checkpoint)
    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Student encoder checkpoint did not match architecture: "
            f"missing={missing}, unexpected={unexpected}"
        )
    student.eval()
    return student


def load_cpsam_student_head(path: str | Path, net: nn.Module, device: torch.device | str = "cpu") -> bool:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    head_state = checkpoint.get("student_head_state_dict")
    if not head_state:
        return False
    if "out.weight" not in head_state or "out.bias" not in head_state:
        raise RuntimeError("Distilled checkpoint head is missing CPSAM readout weights.")

    ps = int(getattr(net, "ps", 8))
    out_weight = head_state["out.weight"]
    nout = int(out_weight.shape[0] // (ps**2))
    net.nout = nout
    net.out = nn.Conv2d(256, out_weight.shape[0], kernel_size=1).to(device)
    missing, unexpected = net.out.load_state_dict(
        {
            "weight": head_state["out.weight"],
            "bias": head_state["out.bias"],
        },
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Distilled checkpoint head did not match CPSAM readout: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if "W2" in head_state:
        net.W2 = nn.Parameter(head_state["W2"].to(device), requires_grad=False)
    net.to(device)
    return True
