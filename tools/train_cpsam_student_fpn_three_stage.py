"""Three-stage CPSAM distillation with a multiscale MobileNetV3-large FPN student."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from cellpose.cpsam_student import MobileNetV3LargeFPNEncoderStudent
from tools.train_cpsam_encoder_distill import (
    CPSAMReadoutHead,
    build_teacher,
    discover_frame_refs,
    load_student_checkpoint,
)
from tools.train_cpsam_student_three_stage import (
    discover_supervised_pairs,
    make_distill_loader,
    make_supervised_loader,
    run_feature_stage,
    run_output_stage,
    run_supervised_stage,
)


@dataclass
class FPNThreeStageConfig:
    data_dir: str | None
    file_list: str | None
    train_root_dirs: tuple[str, ...]
    output_dir: str
    cpsam_model: str
    tile_size: int
    steps_per_epoch: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    student_backbone: str
    student_width: int
    student_depth: int
    init_student: str | None
    resume_student: str | None
    sampling_mode: str
    tile_overlap: float
    augment_scale_range: float
    augment_rotate: bool
    augment_flip: bool
    seed: int
    teacher_device: str
    train_device: str
    amp: bool
    normalize_percentiles: tuple[float, float]
    feature_loss_weight: float
    output_loss_weight: float
    train_cpsam_head: bool
    stage1_epochs: int
    stage2_epochs: int
    stage3_epochs: int
    stage1_feature_weight: float
    stage2_feature_weight: float
    stage2_output_weight: float
    stage2_flow_direction_weight: float
    stage3_seg_weight: float
    stage3_flow_direction_weight: float
    stage3_flow_device: str
    stage3_flow_cache_dir: str | None
    mobilenet_weights: str
    mobilenet_weights_path: str | None
    fpn_tap_layers: tuple[int, ...]
    fpn_width: int
    fpn_context_dilations: tuple[int, ...]
    profile_batches: int


def build_student(config: FPNThreeStageConfig) -> torch.nn.Module:
    return MobileNetV3LargeFPNEncoderStudent(
        weights=config.mobilenet_weights,
        weights_path=config.mobilenet_weights_path,
        tap_layers=config.fpn_tap_layers,
        out_channels=256,
        target_stride=8,
        fpn_width=config.fpn_width,
        context_dilations=config.fpn_context_dilations,
    )


def build_student_head_from_teacher(teacher, device: torch.device) -> CPSAMReadoutHead:
    return CPSAMReadoutHead(teacher.out, teacher.W2, teacher.ps).to(device)


def train(config: FPNThreeStageConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    refs = discover_frame_refs(config.data_dir, config.file_list, config.train_root_dirs)
    print(f"discovered {len(refs)} image/frame references")
    if not refs:
        raise ValueError("No training images discovered.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fpn_three_stage_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    teacher_device = torch.device(config.teacher_device)
    train_device = torch.device(config.train_device)
    teacher = build_teacher(config.cpsam_model, teacher_device, config.tile_size)
    student = build_student(config).to(train_device)
    student_head = build_student_head_from_teacher(teacher, train_device)

    if config.init_student:
        print(f"initializing FPN student from {config.init_student}")
        load_student_checkpoint(config.init_student, student, student_head=student_head, device=train_device, strict=False)
    if config.resume_student:
        print(f"resuming FPN student/head from {config.resume_student}")
        load_student_checkpoint(config.resume_student, student, student_head=student_head, device=train_device, strict=False)

    distill_loader = make_distill_loader(config, refs)
    if config.stage1_epochs > 0:
        run_feature_stage(config, distill_loader, teacher, student, student_head, train_device, teacher_device)
    if config.stage2_epochs > 0:
        run_output_stage(config, distill_loader, teacher, student, student_head, train_device, teacher_device)

    pairs = discover_supervised_pairs(refs)
    print(f"discovered {len(pairs)} supervised image/mask pairs")
    if config.stage3_epochs > 0:
        if not pairs:
            print("stage3 requested, but no matching *_masks.tif/_masks.tiff/_masks.png or *_seg.npy labels were found; skipping supervised fine-tuning")
        else:
            supervised_loader = make_supervised_loader(config, pairs)
            run_supervised_stage(config, supervised_loader, student, student_head, train_device)


def parse_args(argv: Sequence[str] | None = None) -> FPNThreeStageConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--file-list", default=None)
    parser.add_argument("--train-root-dirs", nargs="+", default=())
    parser.add_argument("--output-dir", default="distilled_cpsam_encoder_fpn_three_stage")
    parser.add_argument("--cpsam-model", default="cpsam")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--init-student", default=None)
    parser.add_argument("--resume-student", default=None)
    parser.add_argument("--sampling-mode", default="cellpose-random", choices=("cellpose-random", "overlap-grid"))
    parser.add_argument("--tile-overlap", type=float, default=0.1)
    parser.add_argument("--augment-scale-range", type=float, default=1.5)
    parser.add_argument("--no-augment-rotate", action="store_true")
    parser.add_argument("--no-augment-flip", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--normalize-percentiles", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--stage1-epochs", type=int, default=300)
    parser.add_argument("--stage2-epochs", type=int, default=120)
    parser.add_argument("--stage3-epochs", type=int, default=100)
    parser.add_argument("--stage1-feature-weight", type=float, default=1.0)
    parser.add_argument("--stage2-feature-weight", type=float, default=0.05)
    parser.add_argument("--stage2-output-weight", type=float, default=5.0)
    parser.add_argument("--stage2-flow-direction-weight", type=float, default=1.0)
    parser.add_argument("--stage3-seg-weight", type=float, default=1.0)
    parser.add_argument("--stage3-flow-direction-weight", type=float, default=0.5)
    parser.add_argument(
        "--stage3-flow-device",
        default="cpu",
        help="Device used to precompute supervised label flow targets before stage 3, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--stage3-flow-cache-dir",
        default=None,
        help="Directory for cached supervised flow .npy files. Defaults to <output-dir>/supervised_flow_cache.",
    )
    parser.add_argument("--mobilenet-weights", default="imagenet", choices=("imagenet", "none"))
    parser.add_argument("--mobilenet-weights-path", default=None)
    parser.add_argument(
        "--fpn-tap-layers",
        type=int,
        nargs="+",
        default=(6, 12, 16),
        help="MobileNetV3-large feature layers to fuse. Defaults to stride-8/16/32 context layers.",
    )
    parser.add_argument("--fpn-width", type=int, default=192)
    parser.add_argument("--fpn-context-dilations", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--profile-batches", type=int, default=0, help="Print timing breakdown for the first N batches of each stage.")
    args = parser.parse_args(argv)
    if args.data_dir is None and args.file_list is None and not args.train_root_dirs:
        parser.error("Provide --data-dir, --file-list, and/or --train-root-dirs.")
    return FPNThreeStageConfig(
        data_dir=args.data_dir,
        file_list=args.file_list,
        train_root_dirs=tuple(args.train_root_dirs),
        output_dir=args.output_dir,
        cpsam_model=args.cpsam_model,
        tile_size=args.tile_size,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        student_backbone="mobilenet-v3-large-fpn",
        student_width=0,
        student_depth=0,
        init_student=args.init_student,
        resume_student=args.resume_student,
        sampling_mode=args.sampling_mode,
        tile_overlap=args.tile_overlap,
        augment_scale_range=args.augment_scale_range,
        augment_rotate=not args.no_augment_rotate,
        augment_flip=not args.no_augment_flip,
        seed=args.seed,
        teacher_device=args.teacher_device,
        train_device=args.train_device,
        amp=bool(args.amp),
        normalize_percentiles=tuple(args.normalize_percentiles),
        feature_loss_weight=args.stage2_feature_weight,
        output_loss_weight=args.stage2_output_weight,
        train_cpsam_head=True,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        stage1_feature_weight=args.stage1_feature_weight,
        stage2_feature_weight=args.stage2_feature_weight,
        stage2_output_weight=args.stage2_output_weight,
        stage2_flow_direction_weight=args.stage2_flow_direction_weight,
        stage3_seg_weight=args.stage3_seg_weight,
        stage3_flow_direction_weight=args.stage3_flow_direction_weight,
        stage3_flow_device=args.stage3_flow_device,
        stage3_flow_cache_dir=args.stage3_flow_cache_dir,
        mobilenet_weights=args.mobilenet_weights,
        mobilenet_weights_path=args.mobilenet_weights_path,
        fpn_tap_layers=tuple(args.fpn_tap_layers),
        fpn_width=args.fpn_width,
        fpn_context_dilations=tuple(args.fpn_context_dilations),
        profile_batches=args.profile_batches,
    )


def main(argv: Sequence[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
