import numpy as np


def discard_class_zero_instances(masks, classes):
    """Remove class-0 instances and compact remaining mask IDs and classes.

    ``classes`` is indexed by mask ID. A missing class entry is treated as
    class 0 so malformed/unclassified predictions cannot become GUI objects.
    Non-semantic results (``classes is None``) are returned unchanged.
    """
    if masks is None or classes is None:
        return masks, classes

    masks_arr = np.asarray(masks)
    classes_arr = np.asarray(classes, dtype=np.int16)
    filtered = np.zeros_like(masks_arr)
    kept_classes = [0]
    next_id = 1

    for old_id in np.unique(masks_arr[masks_arr > 0]).astype(np.int64):
        class_id = int(classes_arr[old_id]) if old_id < len(classes_arr) else 0
        if class_id <= 0:
            continue
        filtered[masks_arr == old_id] = next_id
        kept_classes.append(class_id)
        next_id += 1

    return filtered, np.asarray(kept_classes, dtype=np.int16)


def build_classes_map_from_masks(masks, classes):
    """Build per-pixel class map from instance masks + per-instance class vector."""
    if masks is None or classes is None:
        return None
    try:
        masks_i64 = np.asarray(masks).astype(np.int64, copy=False)
        classes_arr = np.asarray(classes)
        out = np.zeros_like(masks_i64, dtype=np.int64)
        valid = (masks_i64 >= 0) & (masks_i64 < int(len(classes_arr)))
        if np.any(valid):
            out[valid] = np.rint(classes_arr[masks_i64[valid]]).astype(np.int64, copy=False)
        return out
    except Exception:
        return None


def sanitize_class_map(class_map, masks=None, classes=None, class_names=None):
    """
    Normalize/validate class map with shared local+remote behavior.
    - accepts per-pixel class map or mask-id map
    - remaps mask-id map to classes[] when detected
    - clips to inferred semantic max class
    - enforces 2D shape and masks shape match when masks provided
    """
    if class_map is None:
        return None
    try:
        class_map = np.squeeze(class_map)
    except Exception:
        return None
    if getattr(class_map, "ndim", 0) != 2:
        return None

    try:
        class_map = np.rint(class_map).astype(np.int64, copy=False)
    except Exception:
        return None

    max_class = None
    if class_names is not None:
        try:
            max_class = max(0, int(len(class_names) - 1))
        except Exception:
            max_class = None
    if classes is not None:
        try:
            max_class = max(max_class or 0, int(np.max(classes)))
        except Exception:
            max_class = max_class
    if max_class is None:
        try:
            max_class = int(np.max(class_map))
        except Exception:
            max_class = 0

    if masks is not None and classes is not None:
        try:
            masks_i64 = np.asarray(masks).astype(np.int64, copy=False)
            # Out-of-range implies this is probably a mask-id map.
            if int(np.max(class_map)) > int(max_class):
                remapped = build_classes_map_from_masks(masks_i64, classes)
                if remapped is not None:
                    class_map = remapped
            else:
                # Heuristic: if class_map matches mask ids on most mask pixels, remap.
                mask_pixels = masks_i64 > 0
                if np.any(mask_pixels):
                    same = class_map[mask_pixels] == masks_i64[mask_pixels]
                    if float(np.mean(same)) > 0.9:
                        remapped = build_classes_map_from_masks(masks_i64, classes)
                        if remapped is not None:
                            class_map = remapped
        except Exception:
            pass

    class_map = np.clip(class_map, 0, int(max_class))
    if masks is not None:
        try:
            if tuple(class_map.shape) != tuple(np.asarray(masks).shape):
                return None
        except Exception:
            return None
    return class_map

