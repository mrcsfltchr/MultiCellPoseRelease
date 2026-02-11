import logging
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from guv_app.plugins.interface import AnalysisPlugin
from guv_app.plugins.validator import validate_visualization_mask

_logger = logging.getLogger(__name__)


class ObjectClustersPlugin(AnalysisPlugin):
    """
    Groups same-class object masks into clusters based on shared interface length.

    Two masks are connected when they are touching neighbors and the number of
    touching pixel pairs between them is >= interface_length_min.
    """

    @property
    def name(self) -> str:
        return "Object Clusters"

    def get_parameter_definitions(self):
        return {
            "interface_length_min": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 10000,
                "label": "Min Interface Length (px)",
                "help": "Minimum shared boundary length required to consider two masks connected.",
            },
            "min_cluster_size": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 10000,
                "label": "Min Cluster Size",
                "help": "Minimum number of masks to report a cluster.",
            },
        }

    def run(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        del image  # Not required for topology-only analysis.
        if masks is None:
            return pd.DataFrame()

        interface_length_min = int(kwargs.get("interface_length_min", 5))
        min_cluster_size = int(kwargs.get("min_cluster_size", 2))
        masks2d = _to_2d_masks(masks)
        if masks2d is None or masks2d.max() <= 0:
            return pd.DataFrame()

        pairs = _compute_interface_lengths(masks2d)
        if not pairs:
            return pd.DataFrame()

        adjacency = _build_same_class_graph(
            pairs=pairs,
            classes=classes,
            interface_length_min=interface_length_min,
        )
        if not adjacency:
            return pd.DataFrame()

        clusters = _connected_components(adjacency)
        rows = []
        cluster_idx = 0
        for comp in clusters:
            if len(comp) < min_cluster_size:
                continue
            cluster_idx += 1
            comp_sorted = sorted(comp)
            class_id = _safe_class(classes, comp_sorted[0])
            internal_interface_sum = _sum_internal_interface(pairs, comp_sorted)
            rows.append(
                {
                    "cluster_id": cluster_idx,
                    "class_id": class_id,
                    "cluster_size": len(comp_sorted),
                    "mask_ids": ";".join(str(m) for m in comp_sorted),
                    "total_interface_px": int(internal_interface_sum),
                }
            )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def visualize(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> np.ndarray:
        del image
        masks2d = _to_2d_masks(masks)
        if masks2d is None or masks2d.max() <= 0:
            return np.zeros_like(masks2d if masks2d is not None else np.zeros((1, 1), dtype=np.int32))

        interface_length_min = int(kwargs.get("interface_length_min", 5))
        min_cluster_size = int(kwargs.get("min_cluster_size", 2))

        pairs = _compute_interface_lengths(masks2d)
        adjacency = _build_same_class_graph(
            pairs=pairs,
            classes=classes,
            interface_length_min=interface_length_min,
        )
        clusters = _connected_components(adjacency)

        out = np.zeros_like(masks2d, dtype=np.int32)
        for comp in clusters:
            if len(comp) < min_cluster_size:
                continue
            ids = np.array(list(comp), dtype=np.int32)
            out[np.isin(masks2d, ids)] = masks2d[np.isin(masks2d, ids)]

        validate_visualization_mask(out, masks2d)
        return out


def _to_2d_masks(masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 2:
        return arr.astype(np.int32, copy=False)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0].astype(np.int32, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.int32, copy=False)
    _logger.warning("Object Clusters: unsupported mask shape %s; expected (H,W) or singleton-3D.", arr.shape)
    return None


def _compute_interface_lengths(masks2d: np.ndarray) -> Dict[Tuple[int, int], int]:
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    # Horizontal neighbors
    left = masks2d[:, :-1]
    right = masks2d[:, 1:]
    diff = (left != right) & (left > 0) & (right > 0)
    if np.any(diff):
        a = left[diff].astype(np.int32, copy=False)
        b = right[diff].astype(np.int32, copy=False)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        for i in range(lo.size):
            pair_counts[(int(lo[i]), int(hi[i]))] += 1

    # Vertical neighbors
    up = masks2d[:-1, :]
    down = masks2d[1:, :]
    diff = (up != down) & (up > 0) & (down > 0)
    if np.any(diff):
        a = up[diff].astype(np.int32, copy=False)
        b = down[diff].astype(np.int32, copy=False)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        for i in range(lo.size):
            pair_counts[(int(lo[i]), int(hi[i]))] += 1

    return dict(pair_counts)


def _safe_class(classes: np.ndarray, mask_id: int) -> int:
    if classes is None:
        return 0
    if mask_id < 0 or mask_id >= len(classes):
        return 0
    return int(classes[mask_id])


def _build_same_class_graph(
    pairs: Dict[Tuple[int, int], int],
    classes: np.ndarray,
    interface_length_min: int,
) -> Dict[int, Set[int]]:
    graph: Dict[int, Set[int]] = defaultdict(set)
    for (m1, m2), length in pairs.items():
        if length < interface_length_min:
            continue
        c1 = _safe_class(classes, m1)
        c2 = _safe_class(classes, m2)
        if c1 <= 0 or c1 != c2:
            continue
        graph[m1].add(m2)
        graph[m2].add(m1)
    return dict(graph)


def _connected_components(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    components: List[Set[int]] = []
    visited: Set[int] = set()

    for start in graph:
        if start in visited:
            continue
        comp: Set[int] = set()
        q = deque([start])
        visited.add(start)
        while q:
            node = q.popleft()
            comp.add(node)
            for nbr in graph.get(node, ()):
                if nbr not in visited:
                    visited.add(nbr)
                    q.append(nbr)
        components.append(comp)
    return components


def _sum_internal_interface(pairs: Dict[Tuple[int, int], int], members: List[int]) -> int:
    member_set = set(members)
    total = 0
    for (m1, m2), length in pairs.items():
        if m1 in member_set and m2 in member_set:
            total += int(length)
    return total
