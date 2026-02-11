import os
import random
import logging
import glob

import numpy as np
from cellpose import io as cellpose_io
from cellpose.semantic_label_utils import (
    build_classes_map_from_masks,
    sanitize_class_map,
)

_logger = logging.getLogger(__name__)


class TrainingDatasetService:
    def _sanitize_class_map(self, class_map, masks=None, classes=None, class_names=None):
        return sanitize_class_map(
            class_map,
            masks=masks,
            classes=classes,
            class_names=class_names,
        )

    def list_training_images(self, folder_path, mask_filter="_masks", look_one_level_down=False):
        if not folder_path:
            return []
        return cellpose_io.get_image_files(
            folder_path, mask_filter, look_one_level_down=look_one_level_down
        )

    def split_train_test(self, image_files, test_ratio, seed=None):
        if not image_files or test_ratio <= 0:
            return list(image_files), []
        rng = random.Random(seed)
        files = list(image_files)
        rng.shuffle(files)
        n_total = len(files)
        n_test = int(round(n_total * test_ratio))
        n_test = min(max(1, n_test), max(1, n_total - 1))
        test_files = files[:n_test]
        train_files = files[n_test:]
        return train_files, test_files

    def pair_images_with_labels(self, image_files, mask_suffix="_seg.npy", expand_series=False):
        train_files = []
        label_files = []
        missing = []
        for image_path in image_files:
            base, _ = os.path.splitext(image_path)
            label_path = f"{base}{mask_suffix}"
            if os.path.exists(label_path):
                train_files.append(image_path)
                label_files.append(label_path)
            else:
                if expand_series:
                    # Check for multi-series/timepoint labels like __S1, __S1_T2, etc.
                    glob_pattern = f"{base}__*{mask_suffix}"
                    matches = sorted(glob.glob(glob_pattern))
                    if matches:
                        for seg_path in matches:
                            seg_base = seg_path[: -len(mask_suffix)]
                            if not seg_base.startswith(base + "__"):
                                continue
                            frame_suffix = seg_base[len(base) + 2 :]
                            if not frame_suffix:
                                continue
                            train_files.append(f"{image_path}::{frame_suffix}")
                            label_files.append(seg_path)
                        continue
                missing.append(image_path)
        return train_files, label_files, missing

    def validate_segmentation(self, seg_path):
        issues = []
        class_map = None
        if not os.path.exists(seg_path):
            return False, ["missing seg file"], None
        try:
            dat = np.load(seg_path, allow_pickle=True).item()
        except Exception as exc:
            return False, [f"failed to read seg file: {exc}"], None
        masks = dat.get("masks")
        if masks is None:
            issues.append("missing masks in seg file")
            return False, issues, None
        masks = np.squeeze(masks)
        if masks.ndim != 2:
            issues.append("masks are not 2D")
            return False, issues, None
        class_map = dat.get("classes_map")
        if class_map is not None:
            class_map = self._sanitize_class_map(
                class_map,
                masks=masks,
                classes=dat.get("classes"),
                class_names=dat.get("class_names"),
            )
            if class_map is None:
                issues.append("classes_map shape mismatch")
        classes = dat.get("classes")
        if classes is not None:
            try:
                if int(masks.max()) >= len(classes):
                    issues.append("classes array too small for mask ids")
            except Exception:
                issues.append("invalid classes array")
        return len(issues) == 0, issues, class_map

    def validate_training_pairs(self, image_files, label_files):
        valid_images = []
        valid_labels = []
        invalid = []
        for image_path, label_path in zip(image_files, label_files):
            ok, issues, _ = self.validate_segmentation(label_path)
            if ok:
                valid_images.append(image_path)
                valid_labels.append(label_path)
            else:
                invalid.append((image_path, label_path, issues))
        return valid_images, valid_labels, invalid

    def load_local_sets(self, image_files):
        if not image_files:
            return [], [], [], [], []
        train_data = []
        train_labels = []
        train_files = []
        class_maps = []
        invalid = []
        for image_path in image_files:
            if "::" in image_path:
                base_filename, frame_id = image_path.split("::", 1)
            else:
                base_filename, frame_id = image_path, None
            base = os.path.splitext(base_filename)[0]
            seg_path = f"{base}_seg.npy"
            if frame_id:
                seg_path = f"{base}__{frame_id}_seg.npy"
            seg_paths = []
            if os.path.exists(seg_path):
                seg_paths = [(seg_path, frame_id)]
            elif frame_id is None:
                glob_pattern = f"{base}__*_seg.npy"
                matches = sorted(glob.glob(glob_pattern))
                for seg_match in matches:
                    seg_base = seg_match[: -len("_seg.npy")]
                    if not seg_base.startswith(base + "__"):
                        continue
                    frame_suffix = seg_base[len(base) + 2 :]
                    if frame_suffix:
                        seg_paths.append((seg_match, frame_suffix))

            for seg_path_item, seg_frame_id in seg_paths:
                ok, issues, class_map = self.validate_segmentation(seg_path_item)
                if not ok:
                    if seg_frame_id:
                        image_ref = f"{base_filename}::{seg_frame_id}"
                    else:
                        image_ref = base_filename
                    invalid.append((image_ref, seg_path_item, issues))
                    continue
                dat = np.load(seg_path_item, allow_pickle=True).item()
                masks = np.squeeze(dat.get("masks"))
                if seg_frame_id:
                    frame = cellpose_io.read_image_frame(base_filename, seg_frame_id)
                    data = frame.array if frame is not None else None
                else:
                    data = cellpose_io.imread(base_filename)
                if data is None:
                    continue
                if class_map is None:
                    classes = dat.get("classes")
                    if classes is not None:
                        class_map = build_classes_map_from_masks(masks, classes)
                class_map = self._sanitize_class_map(
                    class_map,
                    masks=masks,
                    classes=dat.get("classes"),
                    class_names=dat.get("class_names"),
                )
                if seg_frame_id:
                    train_files.append(f"{base_filename}::{seg_frame_id}")
                else:
                    train_files.append(base_filename)
                train_data.append(data)
                train_labels.append(masks)
                class_maps.append(class_map)
        return train_data, train_labels, train_files, class_maps, invalid
