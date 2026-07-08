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


def compute_class_weights_from_class_maps(class_maps, nclasses=None,
                                          include_background=True, normalize=True):
    """
    Compute inverse-frequency class weights from class maps.

    Returns a weight vector including background at index 0.

    When ``include_background`` is True (default) the background class (id 0)
    is part of the inverse-frequency calculation, so its typically large pixel
    count gives it a correspondingly small weight and it no longer dominates
    the mean cross-entropy. This is what lets the class head receive real
    gradient to separate the foreground classes. Set ``include_background`` to
    False to restore the legacy behaviour where background weight is fixed at
    1.0. When ``normalize`` is True the weights are rescaled so their mean is
    ~1, keeping the overall class-loss magnitude stable regardless of the
    number of classes.
    """
    if not class_maps:
        return None
    try:
        cleaned = []
        for cmap in class_maps:
            if cmap is None:
                continue
            cmap = np.squeeze(cmap)
            if getattr(cmap, "ndim", 0) != 2:
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

        n_present = global_max + 1
        # Accumulate global pixel counts per class (index 0 == background) so
        # frequencies reflect the whole dataset, not a per-image average.
        counts = np.zeros(n_present, dtype=np.float64)
        for cmap in cleaned:
            counts += np.bincount(cmap.ravel(), minlength=n_present)[:n_present]

        total = counts.sum()
        if total <= 0:
            return None
        freq = counts / total

        # Inverse frequency; classes with no pixels get a neutral weight of 1.0.
        inv = np.ones(n_present, dtype=np.float32)
        nonzero = freq > 0
        inv[nonzero] = (1.0 / freq[nonzero]).astype(np.float32)
        if not include_background:
            inv[0] = 1.0

        if nclasses is None:
            nclasses = n_present
        nclasses = int(max(1, nclasses))
        weights = np.ones(nclasses, dtype=np.float32)
        fill_len = min(n_present, nclasses)
        weights[:fill_len] = inv[:fill_len]

        if normalize and weights.sum() > 0:
            weights *= float(nclasses) / float(weights.sum())
        return weights
    except Exception:
        return None

