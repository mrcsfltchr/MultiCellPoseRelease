# Make Custom Plugin Skill

Use this skill when asked to create or modify analysis plugins for MultiCellPose.

## Goal

Produce a valid plugin Python file under `guv_app/plugins/` that implements `AnalysisPlugin` in `guv_app/plugins/interface.py`, plus optional visualization and parameter schema.

## Inputs You Should Ask For

1. Plugin name (display name).
2. Exact analysis objective.
3. Required outputs (columns in result DataFrame).
4. Hyperparameters with defaults/ranges.
5. Whether visualization is required.

## Required Interface Contract

Implement:

- `name` property returning display string.
- `run(image, masks, classes=None, **kwargs) -> pandas.DataFrame`.

Optional:

- `get_parameter_definitions()` returns dict schema.
- `visualize(...) -> np.ndarray | None`.

## Implementation Workflow

1. Inspect `guv_app/plugins/interface.py`.
2. Mirror style from `guv_app/plugins/basic_stats.py` and `guv_app/plugins/perimeter_intensity.py`.
3. Handle mask shape robustness:
   - Prefer `(H, W)` masks.
   - Support singleton 3D masks `(1, H, W)` when practical.
4. Validate visualization masks with:
   - `from guv_app.plugins.validator import validate_visualization_mask`
5. Keep plugin side-effect free:
   - Return a DataFrame, do not write files.
6. Register by placement only:
   - Place file in `guv_app/plugins/`; discovery is automatic.

## Validation Step (Mandatory)

Run:

```powershell
python scripts/validate_plugins.py
```

If failing, fix plugin and re-run.

## Output Checklist

- New plugin file in `guv_app/plugins/`.
- Parameter definitions documented in code.
- `run()` returns stable DataFrame schema.
- `visualize()` returns valid mask or `None`.
- Validator passes.
