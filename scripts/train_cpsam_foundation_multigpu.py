#!/usr/bin/env python
"""
Multi-GPU version of CPSAM foundation training script.

This script supports two approaches for multi-GPU training:
1. DataParallel (simple, good for 2-4 GPUs on single machine)
2. DistributedDataParallel (scalable, best for 4+ GPUs or multi-node)

Example usage - DataParallel (simple):
    python scripts/train_cpsam_foundation_multigpu.py \
        --train-dir data/foundation_cpsam/train \
        --foundation-training \
        --multi-gpu dataparallel \
        --epochs 300 \
        --batch-size 40

Example usage - DistributedDataParallel (advanced):
    # Launch with torchrun (recommended)
    torchrun --nproc_per_node=4 scripts/train_cpsam_foundation_multigpu.py \
        --train-dir data/foundation_cpsam/train \
        --foundation-training \
        --multi-gpu ddp \
        --epochs 300 \
        --batch-size 40

    # Or with python -m torch.distributed.launch (older PyTorch)
    python -m torch.distributed.launch --nproc_per_node=4 \
        scripts/train_cpsam_foundation_multigpu.py \
        --train-dir data/foundation_cpsam/train \
        --foundation-training \
        --multi-gpu ddp \
        --epochs 300 \
        --batch-size 40
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from cellpose import train as cellpose_train
from cellpose.models import CellposeModel
from guv_app.data_models.configs import TrainingConfig
from guv_app.services import training_service as training_service_module
from guv_app.services.training_dataset_service import TrainingDatasetService
from guv_app.services.training_service import TrainingService

logger = logging.getLogger("train_cpsam_foundation_multigpu")


def setup_distributed(backend="nccl"):
    """Initialize distributed training process group.

    Returns:
        rank: Process rank (GPU ID)
        world_size: Total number of GPUs
        is_distributed: Whether running in distributed mode
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        logger.info(f"Initialized DDP: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, True
    else:
        return 0, 1, False


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_for_multigpu(net, multi_gpu_mode, device):
    """Wrap model for multi-GPU training.

    Args:
        net: PyTorch model
        multi_gpu_mode: 'dataparallel', 'ddp', or None
        device: Primary device

    Returns:
        wrapped_net: Model wrapped for multi-GPU
        is_wrapped: Whether model was wrapped
    """
    if multi_gpu_mode is None or not torch.cuda.is_available():
        return net, False

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        logger.warning(f"Only {gpu_count} GPU(s) available, multi-GPU disabled")
        return net, False

    if multi_gpu_mode == "dataparallel":
        logger.info(f"Using DataParallel with {gpu_count} GPUs")
        net = DataParallel(net)
        return net, True

    elif multi_gpu_mode == "ddp":
        rank, world_size, is_distributed = setup_distributed()
        if not is_distributed:
            logger.error("DDP requested but distributed environment not initialized. Use torchrun or torch.distributed.launch")
            sys.exit(1)

        logger.info(f"Using DistributedDataParallel: rank={rank}/{world_size}")
        net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        return net, True

    else:
        logger.error(f"Unknown multi_gpu_mode: {multi_gpu_mode}")
        sys.exit(1)


def _setup_logging(verbose: bool, rank: int = 0) -> None:
    """Setup logging (only rank 0 in distributed mode)."""
    level = logging.DEBUG if verbose else logging.INFO

    # In distributed mode, only rank 0 logs to avoid clutter
    if rank == 0:
        logging.basicConfig(
            level=level,
            format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
            stream=sys.stdout,
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def _discover_labeled(dataset_service: TrainingDatasetService, train_dir: str):
    images = dataset_service.list_training_images(train_dir, look_one_level_down=True)
    data, labels, files, class_maps, invalid = dataset_service.load_local_sets(images)
    if invalid:
        logger.warning("Skipped invalid items: %s", len(invalid))
    return data, labels, files, class_maps


def _split_indices(n_total: int, test_ratio: float, seed: int):
    if n_total <= 1 or test_ratio <= 0:
        return np.arange(n_total, dtype=int), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_total)
    n_test = int(round(n_total * test_ratio))
    n_test = min(max(1, n_test), max(1, n_total - 1))
    return np.sort(order[n_test:]), np.sort(order[:n_test])


def _subset(items, indices):
    if items is None:
        return None
    return [items[i] for i in indices]


def _infer_class_max(class_maps):
    vmax = None
    for cmap in class_maps or []:
        if cmap is None:
            continue
        try:
            cur = int(np.max(cmap))
            vmax = cur if vmax is None else max(vmax, cur)
        except Exception:
            continue
    return vmax


def _estimate_crop_capacity(train_labels, bsize: int) -> int:
    total = 0
    for lbl in train_labels or []:
        try:
            arr = np.asarray(lbl)
            h, w = int(arr.shape[-2]), int(arr.shape[-1])
            total += max(1, int(np.ceil(h / bsize) * np.ceil(w / bsize)))
        except Exception:
            total += 1
    return max(1, total)


def _build_net(base_model: str, class_maps, use_gpu: bool, foundation_training: bool = False, sam_checkpoint: str = None):
    """Build network for training."""
    class_max = _infer_class_max(class_maps)
    if class_max is not None and class_max >= 1 and os.path.basename(str(base_model)) == "cpsam":
        if foundation_training:
            logger.warning(
                "Foundation training mode enabled but class_maps detected. "
                "This may not be appropriate for foundation training. "
                "Proceeding with semantic head initialization anyway."
            )
        return training_service_module._initialize_class_net(nclasses=class_max + 1)

    if foundation_training:
        logger.info("Foundation training mode: initializing from original SAM encoder")
        model = CellposeModel(
            pretrained_model=base_model,
            gpu=use_gpu,
            foundation_training=True,
            sam_checkpoint=sam_checkpoint,
        )
    else:
        model = CellposeModel(pretrained_model=base_model, gpu=use_gpu)
    return model.net


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-GPU CPSAM foundation model training.")
    p.add_argument("--train-dir", default="data/foundation_cpsam/train", help="Canonical collated training dir.")
    p.add_argument("--base-model", default="cpsam", help="Base model id/path.")
    p.add_argument("--foundation-training", action="store_true",
                   help="Initialize from original SAM encoder (not cellpose-SAM weights).")
    p.add_argument("--sam-checkpoint", default=None,
                   help="Path to original SAM checkpoint (e.g., sam_vit_l_0b3195.pth).")
    p.add_argument("--model-name", default="", help="Output model name.")
    p.add_argument("--save-path", default="", help="Model output dir.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=10,
                   help="Total batch size across all GPUs (will be divided by num_gpus)")
    p.add_argument("--bsize", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--scale-range", type=float, default=0.5)
    p.add_argument("--rescale", action="store_true")
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unfreeze-blocks", type=int, default=6, help="Train only last N encoder blocks.")
    p.add_argument("--seg-loss-weight", type=float, default=0.1)
    p.add_argument("--nimg-per-epoch", type=int, default=0, help="0 => auto (25%% of estimated non-overlap crops).")
    p.add_argument("--min-train-masks", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--train-debug", action="store_true")
    p.add_argument("--train-debug-steps", type=int, default=3)
    p.add_argument("--verbose", action="store_true")

    # Multi-GPU options
    p.add_argument("--multi-gpu", type=str, default=None, choices=["dataparallel", "ddp"],
                   help="Multi-GPU mode: 'dataparallel' (simple) or 'ddp' (distributed, requires torchrun)")
    p.add_argument("--local-rank", type=int, default=0,
                   help="Local rank for distributed training (auto-set by torchrun)")

    return p


def main() -> int:
    args = build_parser().parse_args()

    # Setup distributed environment if using DDP
    rank = 0
    world_size = 1
    if args.multi_gpu == "ddp":
        rank, world_size, is_distributed = setup_distributed()
        if not is_distributed:
            logger.error("DDP mode requires launching with torchrun or torch.distributed.launch")
            return 1

    _setup_logging(args.verbose, rank=rank)
    cellpose_train.set_train_debug(args.train_debug, args.train_debug_steps)

    # Only rank 0 discovers and splits data (to avoid race conditions)
    if rank == 0:
        ds = TrainingDatasetService()
        train_data_all, train_labels_all, train_files_all, class_maps_all = _discover_labeled(ds, args.train_dir)
        if not train_files_all:
            logger.error("No labeled data found in %s", args.train_dir)
            return 2

        tr_idx, te_idx = _split_indices(len(train_files_all), args.test_ratio, args.seed)
        train_data = _subset(train_data_all, tr_idx)
        train_labels = _subset(train_labels_all, tr_idx)
        train_files = _subset(train_files_all, tr_idx)
        class_maps = _subset(class_maps_all, tr_idx)
        test_data = _subset(train_data_all, te_idx)
        test_labels = _subset(train_labels_all, te_idx)
        test_files = _subset(train_files_all, te_idx)
        test_class_maps = _subset(class_maps_all, te_idx)

        logger.info("Dataset split: train=%s test=%s", len(train_files), len(test_files))

    # Synchronize data across all processes in DDP mode
    if args.multi_gpu == "ddp":
        # In production, you'd broadcast the data or use shared storage
        # For simplicity, we're assuming all ranks have access to same filesystem
        dist.barrier()  # Wait for rank 0 to finish data loading

        # Re-load on other ranks (not efficient, but simple)
        if rank != 0:
            ds = TrainingDatasetService()
            train_data_all, train_labels_all, train_files_all, class_maps_all = _discover_labeled(ds, args.train_dir)
            tr_idx, te_idx = _split_indices(len(train_files_all), args.test_ratio, args.seed)
            train_data = _subset(train_data_all, tr_idx)
            train_labels = _subset(train_labels_all, tr_idx)
            train_files = _subset(train_files_all, tr_idx)
            class_maps = _subset(class_maps_all, tr_idx)
            test_data = _subset(train_data_all, te_idx)
            test_labels = _subset(train_labels_all, te_idx)
            test_files = _subset(train_files_all, te_idx)
            test_class_maps = _subset(class_maps_all, te_idx)

    crop_capacity = _estimate_crop_capacity(train_labels, args.bsize)
    nimg_per_epoch = args.nimg_per_epoch if args.nimg_per_epoch > 0 else max(len(train_files), int(round(0.25 * crop_capacity)))

    if rank == 0:
        logger.info(
            "nimg_per_epoch=%s (crop_capacity=%s, bsize=%s, auto=%s)",
            nimg_per_epoch,
            crop_capacity,
            args.bsize,
            args.nimg_per_epoch <= 0,
        )

    model_name = args.model_name or f"{Path(str(args.base_model)).name}_foundation_{time.strftime('%Y%m%d_%H%M%S')}"

    # Build network (on appropriate device for DDP)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and not args.cpu else "cpu")
    net = _build_net(
        args.base_model,
        class_maps,
        use_gpu=not args.cpu,
        foundation_training=args.foundation_training,
        sam_checkpoint=args.sam_checkpoint,
    )
    net = net.to(device)

    # Wrap model for multi-GPU
    net, is_wrapped = wrap_model_for_multigpu(net, args.multi_gpu, device)

    if rank == 0:
        if is_wrapped:
            gpu_count = torch.cuda.device_count()
            logger.info(f"Multi-GPU training enabled: {gpu_count} GPUs, mode={args.multi_gpu}")
            logger.info(f"Effective batch size per GPU: {args.batch_size // world_size}")
        else:
            logger.info("Single GPU/CPU training")

    # Adjust batch size for multi-GPU (divide total batch across GPUs)
    batch_size_per_gpu = args.batch_size // world_size if world_size > 1 else args.batch_size

    config = TrainingConfig(
        base_model=args.base_model,
        model_name=model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_epochs=args.epochs,
        batch_size=batch_size_per_gpu,  # Per-GPU batch size
        min_train_masks=args.min_train_masks,
        bsize=args.bsize,
        rescale=args.rescale,
        scale_range=args.scale_range,
        seg_loss_weight=args.seg_loss_weight,
        nimg_per_epoch=nimg_per_epoch,
        use_lora=False,
        unfreeze_blocks=args.unfreeze_blocks,
        save_path=args.save_path or None,
        train_files=train_files,
        test_files=test_files or [],
        train_labels_files=[],
        test_labels_files=[],
    )

    service = TrainingService(net=net)

    def progress_cb(epoch: int, total_epochs: int, train_loss: float, test_loss: float):
        if rank == 0:  # Only rank 0 logs progress
            logger.info(
                "epoch %s/%s train_loss=%.4f test_loss=%.4f",
                epoch + 1,
                total_epochs,
                train_loss,
                test_loss,
            )

    try:
        result = service.start_training(
            config=config,
            progress_callback=progress_cb,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            class_maps=class_maps,
            test_class_maps=test_class_maps,
            flow_labels=None,
            test_flow_labels=None,
        )
    except Exception:
        logger.exception("Training failed.")
        if args.multi_gpu == "ddp":
            cleanup_distributed()
        return 1

    if rank == 0:
        logger.info("Training finished. Model path: %s", result.model_path)

    if args.multi_gpu == "ddp":
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
