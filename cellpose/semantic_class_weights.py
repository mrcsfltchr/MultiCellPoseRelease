import numpy as np


def infer_semantic_nclasses_from_net(net):
    """Infer number of semantic classes (including background) from model output head."""
    try:
        ps = int(getattr(net, "ps", 8))
        out_ch = int(net.out.weight.shape[0])
        nout = max(1, out_ch // (ps**2))
        return max(0, nout - 3)
    except Exception:
        return None


def extract_class_maps_from_labels(labels):
    """
    Extract semantic class maps from flow labels.
    Expected semantic layout is [mask, class_map, flowY, flowX, cellprob] (>=5 channels).
    """
    class_maps = []
    if not labels:
        return class_maps
    for lbl in labels:
        try:
            if getattr(lbl, "ndim", 0) != 3 or lbl.shape[0] < 5:
                continue
            cm = np.squeeze(lbl[1])
            if cm.ndim != 2:
                continue
            if not np.all(np.isfinite(cm)):
                continue
            # Semantic class maps should be integer-like.
            if not np.allclose(cm, np.rint(cm), atol=1e-3):
                continue
            class_maps.append(np.rint(cm).astype(np.int64, copy=False))
        except Exception:
            continue
    return class_maps


def compute_class_weights_from_class_maps(class_maps, nclasses=None):
    """
    Compute inverse-frequency class weights from class maps.
    Returns weight vector including background index 0.
    """
    if not class_maps:
        return None
    try:
        cleaned = []
        for cmap in class_maps:
            if cmap is None:
                continue
            cmap = np.squeeze(cmap)
            if cmap.ndim != 2:
                continue
            cmap = np.rint(cmap).astype(np.int64, copy=False)
            if not np.any(cmap > 0):
                continue
            cleaned.append(cmap)
        if not cleaned:
            return None

        global_max = max(int(np.max(cmap)) for cmap in cleaned)
        if global_max < 1:
            return None

        pimg = []
        for cmap in cleaned:
            counts = np.bincount(cmap.ravel(), minlength=global_max + 1).astype(np.float32)
            counts = counts[1:global_max + 1]
            if counts.size == 0:
                continue
            total = counts.sum() if counts.sum() > 0 else 1.0
            pimg.append(counts / total)
        if not pimg:
            return None

        pclass = np.mean(np.stack(pimg, axis=0), axis=0)
        pclass[pclass == 0] = 1.0
        inv = 1.0 / pclass

        if nclasses is None:
            nclasses = int(global_max + 1)
        nclasses = int(max(1, nclasses))
        weights = np.ones(nclasses, dtype=np.float32)
        fill_len = min(len(inv), max(0, nclasses - 1))
        if fill_len > 0:
            weights[1:1 + fill_len] = inv[:fill_len]
        return weights
    except Exception:
        return None

