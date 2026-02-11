## MultiCellPose:  Trainer + Analyzer

This repository provides a GUI workflow for labeling, remote inference, and training CellposeSAM-based multi-class segmentation models, plus an Analyzer app which allows running custom models and performing custom analysis plugins on detected objects.. It supports local or remote (SSH + gRPC) compute. Note: currently you will need to run the server yourself if you have a remote gpu-enable ssh server.

---

## 1) Installation

### Recommended (PyPI)

#### Step 1: Open Anaconda Prompt (Windows) or terminal (Linux/macOS)

Create and activate a clean environment:

```
conda create -n multicellpose python=3.10 -y
conda activate multicellpose
```

If Conda prompts for Terms of Service acceptance, run:

```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

#### Step 2: Install PyTorch first (choose CPU or GPU)

**GPU users:** check your CUDA version first (Windows):

```
nvidia-smi
```

Look for `CUDA Version: X.Y`. Then install the matching PyTorch wheel:

- CUDA 12.6+:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

- CUDA 12.1:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- CUDA 11.8:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU-only:**

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 3: Install MultiCellPose

```
pip install multicellpose
```

This installs GUI, server/client communication dependencies, and image readers (`.nd2`, `.lif`, `.nrrd`) by default.

#### Step 4: Launch

```
multicellpose-gui
```

Alternative launch command:

```
python -m guv_app.main
```

### From source (developer install)

Use this if you are editing code locally.

#### Step 1: Download the source code

If you already have Git:

```
cd C:\Projects
git clone https://github.com/mrcsfltchr/MultiCellPose.git
cd MultiCellPose
```

If Git is not installed on Windows, install from `https://git-scm.com/download/win` first.

#### Step 2: Create and activate environment

```
conda create -n multicellpose python=3.10 -y
conda activate multicellpose
```

If Conda prompts for Terms of Service acceptance, run:

```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

#### Step 3: Install PyTorch (CPU or GPU)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Adjust wheel index URL for your CUDA version, or use CPU wheel.

#### Step 4: Install repo in editable mode

```
pip install -e ".[all]"
```

#### Step 5: Launch

```
multicellpose-gui
```

This opens the Home Navigator with two apps:
- **Trainer**: labeling + training
- **Analyzer**: batch inference + measurements

Note: the first time you run the app, pretrained model downloads may take several minutes depending on your network.
Critical: ensure your PyTorch CUDA build matches your GPU (older GPUs may require CUDA 11.8/other specific wheels).

---

## 2) Labeling (Trainer)

### Load images
- File -> Load image, or drag-and-drop.
- Supported: `.tif/.tiff/.png/.jpg/.jpeg/.npy/.nd2/.lif/.dax/.nrrd/.flex`.
- `.nd2`/`.lif` support is installed by default with both `pip install multicellpose` and `pip install multicellpose[gui]`.

### Create masks
- Right-click once to start a contour, move the cursor to trace the perimeter, then return to the start to close the loop. You do not need to hold right-click while drawing.
- The GUI fills the contour and assigns a new instance ID.
- Brush size: use `,` and `.` to decrease/increase.
- Delete a single mask: `Ctrl + Left Click`.
- Select masks: left-click on a mask or drag a rectangle to select multiple.
- Assign a class to selected masks: `Shift + Left Click` on a mask.

### Create classes and assign
- In the **Class Management** panel, enter a class name, choose a color, and click **Add**.
- Select the current class from the dropdown before drawing to assign new masks.
- Use **Color by Class** to switch between class colors and instance colors.
- Use **Delete Masks (Lasso)** to remove multiple masks by drawing a closed loop.

### Save labels
- File -> Save Masks (as `*_seg.npy`)
- This saves masks, classes, classes_map, flows, thresholds, and metadata.

---

## Analyzer (Batch Measurements)

The Analyzer app is for running inference across a folder, optionally correcting masks, and exporting measurements (e.g., area and mean intensity) to CSV.

### Load images
- File -> Load image, or drag-and-drop.
- Use the left/right arrow keys to move through a directory after loading.

### Load custom models
- Model -> Add custom model.
- Select a `.pth` or model file in your local models directory.
- The model appears in the model dropdown and can be used for inference.

### Connect to remote
- Remote -> Connect to remote.
- Enter your username and password if server requires authentication.
- Once connected, server models appear in the dropdown and inference will run on the remote server.

### Upload a trained model to the remote
- Remote -> Upload model to server...
- Select the trained model file you downloaded locally (e.g., `.pth`).
- The upload can take a few minutes; a progress bar with ETA will appear.
- After upload, the model will appear in the server model dropdown for inference.

### Run inference
- Select a model and click Run.
- For batch inference, load a directory and run; masks are saved per-image.
 - Note: inference can be slow for image sizes around `(512, 512)` or larger, especially on CPU.

### Correct masks after inference
- Right-click once to draw a new mask (trace a closed contour). You do not need to hold right-click while drawing.
- Delete a single mask with `Ctrl + Left Click`.
- Use **Delete Masks (Lasso)** to remove multiple masks by drawing a closed loop.
- In Analyzer, manual edits are saved to the prediction `.npy` files by default (not the training `*_seg.npy` files).
- Use the Masks menu to promote predicted masks into training labels when needed.

### Run statistics
- Use the Analyzer controls to compute per-object measurements (area, mean intensity, etc.).
- Results are saved to CSV for the full dataset.

---

## 3) Remote connection (SSH + gRPC)

### Server setup (remote machine)

1) Install the repo + deps (same as above).
2) Start the gRPC server:

```
python run_server.py
```

By default it listens on `127.0.0.1:50051` (see `run_server.py`) for secure local-only access.
If you need LAN/WAN exposure, set `server_bind` explicitly and protect it with firewall rules plus strong auth.

### Remote configuration (optional)

You can override server/SSH settings without committing them to Git by creating `remote_config.json` in the repo root (ignored by git). Use `remote_config.example.json` as a template. Supported keys include:

- `server_bind`, `server_storage_root`
- `server_address`
- `ssh_host`, `ssh_port`, `ssh_local_port`, `ssh_remote_port`, `ssh_remote_bind`
- `model_root`, `replay_root`, `replay_sample_size`
- `cpsam_model_path`

### Connect from the GUI

1) Open the GUI.
2) Use Remote -> Connect to remote.
3) Enter your username and password when prompted.
4) The GUI will create an SSH tunnel to the server and fetch model names.

### Manual SSH tunnel (advanced)

```
ssh -L 50051:localhost:50051 username@your.server.host
```

Then in the GUI, connect to `localhost:50051`.

---

## 4) Training

### Start training (Ctrl+T)

1) Load a folder of labeled images (`*_seg.npy`, or masks/flows/classes triplets).
2) Press `Ctrl+T`.
3) Choose a base model and settings.
4) The GUI uploads data to the server (remote mode) or trains locally.
5) Note: training can be slow for large images, especially on CPU.

### What happens during training

- The GUI builds a manifest of labeled items in the current folder.
- Remote mode uploads the manifest + required files to the server.
- The server:
  - validates items
  - computes flows (if needed)
  - initializes the model
  - optionally mixes in replay data from a server-side replay dataset
  - trains and saves artifacts into a `train_jobs` folder

Artifacts include:
- model weights
- training losses (`*_train_losses.npy`)
- metadata (`*_meta.json`)

### Training with semantic classes from cpsam

If `cpsam` is selected as the base model and semantic labels exist, the model head is initialized using the exact logic from `tools/train_inst_seg_fixed.py` so it matches your semantic model architecture (background + classes + flow/cellprob heads).

---

## Remote storage housekeeping

If you hit your remote quota, use:

Remote -> Clear remote training files...

This removes all previous training job files for your user on the server.

---

## Troubleshooting

- **CUDA not available**: install a PyTorch CUDA wheel that matches your GPU.
- **Missing nd2/readlif**: reinstall package dependencies (`pip install --upgrade multicellpose`) and restart.
- **Remote connection fails**: confirm the server is running and port 50051 is reachable.
- **No labels detected**: ensure files are `*_seg.npy` or matching `_masks/_flows/_classes` triplets.

---

## Developer docs

### Image format loader guide

To add a new image format, implement a custom `ImageReader` and register it in `cellpose/io.py`. The GUI and services call the reader through the shared interface, so the app stays format-agnostic.

Requirements:
- Provide `read(...)` and `iter_frames(...)`. If your format supports random access, implement `read_frame(...)` and `get_series_time_info(...)` for fast navigation.
- Populate `ImageMeta` with accurate `axes`, `shape`, and `sizes` when possible.
- Return channel-last arrays (`YXC`) for multi-channel 2D images.
- Use `frame_id` values like `S0`, `S0_T1`, or `T3` for series/time indexing.

Implementation steps:
1) Add a reader class in `cellpose/io.py` (see `_Nd2Reader` and `_LifReader`).
2) Implement `extensions` for your file suffixes.
3) Register it with `register_reader(...)`.
4) If your format can be large, implement `read_frame(...)` and `get_series_time_info(...)` to avoid loading the full file.

Testing:
- Add/extend tests in `tests/test_io_*.py`.
- Validate that arrow-key navigation works for multi-frame files.

### Custom plugin guide

Plugins live under `guv_app/plugins` and follow a standard interface. The Analyzer discovers and registers them at startup.

Requirements:
- Implement the base interface in `guv_app/plugins/interface.py`.
- Provide a unique name, optional config schema, and a `run(...)` method.
- Keep plugins pure: do not mutate GUI state directly. Return results to the controller via the plugin interface.

Implementation steps:
1) Create a new plugin module in `guv_app/plugins/`.
2) Inherit from the plugin base class and implement required methods.
3) If configuration is needed, add a config UI via `guv_app/views/dialogs/plugin_config_dialog.py`.
4) Ensure results are serializable for CSV export.

Testing:
- Add a unit test in `guv_app/tests/` using existing plugin tests as templates.
- Run the Analyzer and verify the plugin appears in the dropdown and produces output.

#### Experimental: LLM plugin-creator skill (in development)

An in-repo LLM skill is available to help scaffold new Analyzer plugins:

- Skill file: `llm_skills/make-custom-plugin/SKILL.md`
- Template: `llm_skills/make-custom-plugin/templates/plugin_template.py`

Status:
- This feature is **experimental** and currently **in development**.
- Generated plugins should always be reviewed manually before use.

Suggested workflow:
1) Use the skill to generate a new plugin module under `guv_app/plugins/`.
2) Confirm it implements the interface in `guv_app/plugins/interface.py`.
3) Validate plugin contracts:
```
python scripts/validate_plugins.py
```
4) Add/adjust tests in `guv_app/tests/` and verify in the Analyzer UI.

Validator utility files:
- `guv_app/plugins/plugin_validator.py`
- `scripts/validate_plugins.py`

#### Using this skill in coding agents

Codex:
1) Keep it project-local (current setup): `llm_skills/make-custom-plugin/`.
2) Or install globally by copying the folder into your Codex skills directory (for example under `$CODEX_HOME/skills/`), then reference it by name in your prompt.
3) When invoking the skill, include the plugin objective, required output columns, and parameter definitions.

Claude coding agents:
1) Claude does not use Codex `SKILL.md` as a guaranteed native standard.
2) Use the same content as an agent instruction file (for example in `CLAUDE.md`) or paste the workflow section into your prompt template.
3) Keep the plugin template file (`llm_skills/make-custom-plugin/templates/plugin_template.py`) in the repo and instruct Claude to generate plugins against `guv_app/plugins/interface.py`.

Validation (recommended for any agent-generated plugin):
```
python scripts/validate_plugins.py
```

---

If you want additional documentation (Analyzer batch measurement workflow, data formats, or deployment notes), ask and I will expand this README.

---

## Acknowledgements

This project builds on the Cellpose project. Please see the Cellpose documentation for additional GUI usage details and background information:
https://www.cellpose.org/ and https://cellpose.readthedocs.io/

---

## HPC cluster install and CLI training (GPU)

Use this when you want to run training jobs non-interactively on a shared GPU cluster.

### 1) Prepare environment on the cluster

From a login node:

1. Load your cluster modules as needed (example):
```
module load anaconda
module load cuda/12.1
```

2. Clone the repo:
```
git clone https://github.com/mrcsfltchr/MultiCellPose.git
cd MultiCellPose
```

3. Create and activate a conda env:
```
conda create -n multicellpose python=3.10 -y
conda activate multicellpose
```

4. If required by your cluster Conda setup, accept Terms of Service:
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

5. Install PyTorch with the correct CUDA wheel for the cluster GPUs:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

6. Install MultiCellPose:
```
pip install -e ".[all]"
```

### 2) Run headless training from CLI

The repo includes a headless trainer:
- `train_headless.py`

Example:
```
python train_headless.py \
  --train-dir /path/to/train_dataset \
  --test-ratio 0.2 \
  --base-model cpsam \
  --model-name cpsam_cluster_run01 \
  --epochs 50 \
  --batch-size 10 \
  --bsize 256 \
  --learning-rate 5e-5 \
  --weight-decay 0.1 \
  --use-lora \
  --lora-blocks 4 \
  --save-path /path/to/output_models
```

Notes:
- Use `--test-dir /path/to/test_dataset` instead of `--test-ratio` if you have a separate test set.
- For non-LoRA fine-tuning, remove `--use-lora` and set `--unfreeze-blocks`.
- Add `--train-debug --train-debug-steps 3` to log per-batch timing/memory diagnostics.

### 3) Example Slurm batch script

Save as `train_job.slurm`:
```
#!/bin/bash
#SBATCH --job-name=multicellpose-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load anaconda
module load cuda/12.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multicellpose

cd /path/to/MultiCellPose

python train_headless.py \
  --train-dir /path/to/train_dataset \
  --test-ratio 0.2 \
  --base-model cpsam \
  --model-name cpsam_slurm_run01 \
  --epochs 50 \
  --batch-size 10 \
  --bsize 256 \
  --use-lora \
  --lora-blocks 4 \
  --save-path /path/to/output_models
```

Submit with:
```
sbatch train_job.slurm
```

## Local/Remote Parity TODO

The project is actively consolidating duplicated local and remote code paths.  
Completed: class-map parsing/sanitization and trainability configuration (LoRA/unfreeze) are now shared.

Remaining parity tasks:
- Unify semantic postprocessing for inference (class-map extraction, resize, per-mask class voting) across:
  - local: `guv_app/services/segmentation_service.py`
  - remote: `cpgrpc/server/services.py`
- Unify dataset/label loading and frame expansion logic across:
  - `guv_app/services/training_dataset_service.py`
  - `guv_app/workers/remote_training_worker.py`
  - `cpgrpc/server/services.py`
- Unify training defaults/parameter canonicalization (normalize params, test split semantics, min mask handling) between:
  - training dialog/controller + local worker
  - remote manifest builder + server trainer
- Reduce local duplicate training loop implementation in `guv_app/services/training_service.py` by reusing shared `cellpose/train.py` entry points where possible.

## PyPI Packaging and Release (Draft)

This repository now includes:
- `pyproject.toml` (PEP 621 metadata + `setuptools_scm`)
- CI packaging checks: `.github/workflows/ci.yml`
- PyPI publish workflow: `.github/workflows/publish-pypi.yml`

### Build locally

From the repo root:
```
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
```

### Windows install test (NVIDIA GPU, CUDA 12.1)

From the repo root, use the helper script:
```
powershell -ExecutionPolicy Bypass -File .\scripts\test_install_windows_cuda121.ps1
```

This script:
- creates a fresh conda env
- accepts conda ToS channels
- installs PyTorch `cu121` wheels
- installs MultiCellPose (`.[all]` by default)
- verifies torch CUDA visibility
- smoke-tests the CLI/import entry points

To test a built wheel instead of editable source:
```
python -m build
powershell -ExecutionPolicy Bypass -File .\scripts\test_install_windows_cuda121.ps1 -PackageSpec ".\dist\multicellpose-<VERSION>-py3-none-any.whl"
```

### Publish from GitHub Actions (Trusted Publishing)

1. Create a PyPI project (name currently set to `multicellpose` in `pyproject.toml`).
2. In PyPI project settings, add a Trusted Publisher:
   - Owner: your GitHub org/user
   - Repository: `mrcsfltchr/MultiCellPose`
   - Workflow: `publish-pypi.yml`
   - Environment: `pypi`
3. In GitHub repository settings, create environment `pypi`.
4. Create a GitHub Release (tag), which triggers publish.

The publish workflow uses OIDC and does not require a PyPI API token secret.
