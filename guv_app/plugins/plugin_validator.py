import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from guv_app.plugins.interface import AnalysisPlugin
from guv_app.plugins.validator import validate_visualization_mask
import guv_app.plugins


@dataclass
class PluginValidationResult:
    module: str
    class_name: str
    plugin_name: str
    valid: bool
    errors: List[str]
    warnings: List[str]


def discover_plugins() -> List[Tuple[str, type]]:
    found: List[Tuple[str, type]] = []
    for _, module_name, _ in pkgutil.iter_modules(guv_app.plugins.__path__, guv_app.plugins.__name__ + "."):
        module = importlib.import_module(module_name)
        for _, obj in inspect.getmembers(module):
            if not inspect.isclass(obj):
                continue
            if not issubclass(obj, AnalysisPlugin):
                continue
            if inspect.isabstract(obj):
                continue
            found.append((module_name, obj))
    return found


def validate_plugin_class(module_name: str, cls: type) -> PluginValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    try:
        plugin = cls()
    except Exception as exc:
        return PluginValidationResult(
            module=module_name,
            class_name=cls.__name__,
            plugin_name=cls.__name__,
            valid=False,
            errors=[f"Instantiation failed: {exc}"],
            warnings=[],
        )

    plugin_name = getattr(plugin, "name", cls.__name__)
    if not isinstance(plugin_name, str) or not plugin_name.strip():
        errors.append("`name` must be a non-empty string.")

    run_sig = inspect.signature(plugin.run)
    for expected in ("image", "masks"):
        if expected not in run_sig.parameters:
            errors.append(f"`run` missing required parameter `{expected}`.")

    # Validate parameter definition schema (if provided)
    try:
        defs = plugin.get_parameter_definitions()
        if defs is None:
            defs = {}
        if not isinstance(defs, dict):
            errors.append("`get_parameter_definitions` must return a dict.")
        else:
            for key, meta in defs.items():
                if not isinstance(key, str):
                    errors.append("Parameter definition keys must be strings.")
                    continue
                if not isinstance(meta, dict):
                    errors.append(f"Parameter `{key}` definition must be a dict.")
                    continue
                ptype = meta.get("type")
                if ptype is not None and ptype not in {"int", "float", "bool", "str", "enum"}:
                    warnings.append(f"Parameter `{key}` has unknown type `{ptype}`.")
    except Exception as exc:
        errors.append(f"Calling `get_parameter_definitions` failed: {exc}")
        defs = {}

    # Dry-run contract checks
    image, masks, classes = _dummy_data()
    try:
        df = plugin.run(image=image, masks=masks, classes=classes, **_defaults_from_defs(defs))
        if df is not None and not isinstance(df, pd.DataFrame):
            errors.append("`run` must return a pandas.DataFrame or None.")
    except Exception as exc:
        errors.append(f"Calling `run` failed: {exc}")

    try:
        viz = plugin.visualize(image=image, masks=masks, classes=classes, **_defaults_from_defs(defs))
        if viz is not None:
            if not isinstance(viz, np.ndarray):
                errors.append("`visualize` must return np.ndarray or None.")
            else:
                validate_visualization_mask(viz, masks)
    except Exception as exc:
        errors.append(f"Calling `visualize` failed: {exc}")

    return PluginValidationResult(
        module=module_name,
        class_name=cls.__name__,
        plugin_name=str(plugin_name),
        valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
    )


def validate_all_plugins() -> List[PluginValidationResult]:
    results = []
    for module_name, cls in discover_plugins():
        results.append(validate_plugin_class(module_name, cls))
    return results


def _defaults_from_defs(defs: Dict) -> Dict:
    defaults = {}
    if not isinstance(defs, dict):
        return defaults
    for key, meta in defs.items():
        if isinstance(meta, dict) and "default" in meta:
            defaults[key] = meta["default"]
    return defaults


def _dummy_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = np.array(
        [
            [0, 1, 1, 0],
            [2, 2, 1, 0],
            [0, 2, 3, 3],
            [0, 0, 3, 0],
        ],
        dtype=np.int32,
    )
    image = np.random.default_rng(0).random((4, 4), dtype=np.float32)
    classes = np.array([0, 1, 1, 2], dtype=np.int32)
    return image, masks, classes
