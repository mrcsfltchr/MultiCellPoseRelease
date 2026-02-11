# Claude Agent Instructions (MultiCellPose)

## Task Type: Create or modify Analyzer plugins

When asked to build a plugin for MultiCellPose, follow this workflow.

### Contract to implement

Plugins must implement `AnalysisPlugin` in:

- `guv_app/plugins/interface.py`

Required:

- `name` property (non-empty string)
- `run(image, masks, classes=None, **kwargs) -> pandas.DataFrame`

Optional:

- `get_parameter_definitions() -> dict`
- `visualize(...) -> np.ndarray | None`

### File placement

- Add new plugins in: `guv_app/plugins/`
- Start from template if useful:
  - `llm_skills/make-custom-plugin/templates/plugin_template.py`

### Implementation rules

1. Keep plugins pure (no GUI state mutation, no file writing).
2. Return stable DataFrame columns from `run`.
3. Handle common mask shapes:
   - `(H, W)`
   - singleton-3D where practical (`(1, H, W)` or `(H, W, 1)`).
4. If visualization is implemented, validate output mask:
   - `from guv_app.plugins.validator import validate_visualization_mask`
5. Use plugin parameter defaults via `get_parameter_definitions`.

### Validation (mandatory)

Run:

```powershell
python scripts/validate_plugins.py
```

Validator files:

- `guv_app/plugins/plugin_validator.py`
- `scripts/validate_plugins.py`

If validation fails, fix and re-run.

