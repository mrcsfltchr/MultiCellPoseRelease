# GUVpose installer

This folder contains a lightweight installer strategy:

- `preflight.py` checks your system and recommends CPU/GPU install.
- `install.ps1` / `install.sh` install dependencies and this repo.
- `repair.ps1` / `repair.sh` re-install dependencies if something breaks.

## 1) Run preflight

Windows (PowerShell):
```
python install/preflight.py
```

macOS/Linux:
```
python3 install/preflight.py
```

## 2) Install

Windows:
```
powershell -ExecutionPolicy Bypass -File install\install.ps1 -Gpu auto
```

macOS/Linux:
```
bash install/install.sh --auto
```

Options:
- `--cpu` forces CPU install
- `--gpu` forces GPU install (requires NVIDIA + CUDA compatible)

## 3) Repair (optional)

Windows:
```
powershell -ExecutionPolicy Bypass -File install\repair.ps1
```

macOS/Linux:
```
bash install/repair.sh
```

## Notes
- GPU installs use PyTorch CUDA wheels. If you need a specific CUDA build,
  override it with `TORCH_INDEX_GPU` on macOS/Linux.
- For `.nd2` and `.lif`, the GUI can prompt to install `nd2`/`readlif`.
  If you accept, restart the app.
