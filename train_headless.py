import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from cellpose import train as cellpose_train
from cellpose.models import CellposeModel
from guv_app.data_models.configs import TrainingConfig
from guv_app.services import training_service as training_service_module
from guv_app.services.training_dataset_service import TrainingDatasetService
from guv_app.services.training_service import TrainingService


logger = logging.getLogger("headless_train")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def _discover_images(
    dataset_service: TrainingDatasetService, folder: str, look_one_level_down: bool
) -> list[str]:
    images = dataset_service.list_training_images(
        folder, look_one_level_down=look_one_level_down
    )
    if not images and not look_one_level_down:
        images = dataset_service.list_training_images(folder, look_one_level_down=True)
    return images


def _load_labeled_dataset(
    dataset_service: TrainingDatasetService, folder: str, look_one_level_down: bool
):
    images = _discover_images(dataset_service, folder, look_one_level_down)
    if not images:
        return [], [], [], []
    data, labels, files, class_maps, _ = dataset_service.load_local_sets(images)
    return data, labels, files, class_maps


def _subset(items, indices):
    if items is None:
        return None
    return [items[i] for i in indices]


def _split_indices(n_total: int, test_ratio: float, seed: int):
    if n_total <= 1 or test_ratio <= 0:
        train_idx = np.arange(n_total, dtype=int)
        return train_idx, np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_total)
    n_test = int(round(n_total * test_ratio))
    n_test = min(max(1, n_test), max(1, n_total - 1))
    test_idx = np.sort(order[:n_test])
    train_idx = np.sort(order[n_test:])
    return train_idx, test_idx


def _infer_class_max(class_maps):
    class_max = None
    for cmap in class_maps or []:
        if cmap is None:
            continue
        try:
            vmax = int(np.max(cmap))
            if class_max is None or vmax > class_max:
                class_max = vmax
        except Exception:
            continue
    return class_max


def _build_model(base_model: str, use_lora: bool, class_maps, use_gpu: bool):
    base_for_init = base_model
    if use_lora and os.path.basename(str(base_for_init)) != "cpsam":
        logger.info(
            "LoRA requested with base model '%s'; initializing from 'cpsam'.", base_for_init
        )
        base_for_init = "cpsam"

    class_max = _infer_class_max(class_maps)
    if class_max is not None and class_max >= 1 and os.path.basename(str(base_for_init)) == "cpsam":
        logger.info("Initializing semantic CPSAM head with %s classes.", class_max + 1)
        return training_service_module._initialize_class_net(nclasses=class_max + 1)

    model = CellposeModel(pretrained_model=base_for_init, gpu=use_gpu)
    return model.net


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless MultiCellPose local training entrypoint."
    )
    parser.add_argument("--train-dir", required=True, help="Folder containing training images/labels.")
    parser.add_argument("--test-dir", default="", help="Optional separate test folder.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio if --test-dir is not provided.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for train/test split.")
    parser.add_argument("--look-one-level-down", action="store_true", help="Also scan one folder level down.")
    parser.add_argument("--base-model", default="cpsam", help="Base model id/path.")
    parser.add_argument("--model-name", default="", help="Output model name (default: auto timestamp).")
    parser.add_argument("--save-path", default="", help="Optional output directory for trained model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--bsize", type=int, default=256, help="Training crop size.")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--scale-range", type=float, default=0.5)
    parser.add_argument("--rescale", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=9,
        help="Number of last encoder blocks to unfreeze in non-LoRA mode.",
    )
    parser.add_argument(
        "--lora-blocks",
        type=int,
        default=None,
        help="Number of last encoder blocks to inject LoRA into (defaults to --unfreeze-blocks).",
    )
    parser.add_argument("--min-train-masks", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU model initialization.")
    parser.add_argument("--train-debug", action="store_true")
    parser.add_argument("--train-debug-steps", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)
    cellpose_train.set_train_debug(args.train_debug, args.train_debug_steps)

    dataset_service = TrainingDatasetService()
    train_data, train_labels, train_files, class_maps = _load_labeled_dataset(
        dataset_service, args.train_dir, args.look_one_level_down
    )

    if not train_files:
        logger.error("No labeled training samples found in: %s", args.train_dir)
        return 2

    if args.test_dir:
        test_data, test_labels, test_files, test_class_maps = _load_labeled_dataset(
            dataset_service, args.test_dir, args.look_one_level_down
        )
        if not test_files:
            logger.warning("No labeled test samples found in: %s", args.test_dir)
    else:
        tr_idx, te_idx = _split_indices(len(train_files), args.test_ratio, args.seed)
        test_data = _subset(train_data, te_idx)
        test_labels = _subset(train_labels, te_idx)
        test_files = _subset(train_files, te_idx)
        test_class_maps = _subset(class_maps, te_idx)
        train_data = _subset(train_data, tr_idx)
        train_labels = _subset(train_labels, tr_idx)
        train_files = _subset(train_files, tr_idx)
        class_maps = _subset(class_maps, tr_idx)

    logger.info(
        "Dataset ready: train=%s samples, test=%s samples",
        len(train_files),
        len(test_files) if test_files else 0,
    )

    model_name = args.model_name or f"{Path(str(args.base_model)).name}_{time.strftime('%Y%m%d_%H%M%S')}"
    net = _build_model(
        base_model=args.base_model,
        use_lora=args.use_lora,
        class_maps=class_maps,
        use_gpu=not args.cpu,
    )

    target_blocks = args.unfreeze_blocks
    lora_blocks = args.lora_blocks
    if args.use_lora:
        if lora_blocks is None:
            lora_blocks = args.unfreeze_blocks
        target_blocks = lora_blocks
    logger.info(
        "Using target encoder blocks: %s (use_lora=%s)",
        target_blocks,
        bool(args.use_lora),
    )

    config = TrainingConfig(
        base_model=args.base_model,
        model_name=model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        min_train_masks=args.min_train_masks,
        bsize=args.bsize,
        rescale=args.rescale,
        scale_range=args.scale_range,
        use_lora=args.use_lora,
        lora_blocks=lora_blocks,
        unfreeze_blocks=target_blocks,
        save_path=args.save_path or None,
        train_files=train_files,
        test_files=test_files or [],
        train_labels_files=[],
        test_labels_files=[],
    )

    service = TrainingService(net=net)

    def progress_cb(epoch: int, total_epochs: int, train_loss: float, test_loss: float):
        logger.info(
            "Progress: epoch %s/%s train_loss=%.4f test_loss=%.4f",
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
        logger.exception("Headless training failed.")
        return 1

    logger.info("Training finished. Model path: %s", result.model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
