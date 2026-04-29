# FPN CPSAM Student PBS Jobs

These jobs run the three FPN distillation phases as separate PBS submissions:

- Stage 1: 300 epochs of CPSAM encoder feature matching.
- Stage 2: 200 epochs of output/flow distillation from the CPSAM head.
- Stage 3: 300 epochs of supervised mask fine-tuning.

Each job follows the same cluster conventions as `train_foundation_4gpu.pbs`:
`$HOME/MultiCellPose` as the repository, `$HOME/FoundationTrain` as the
training root, explicit miniconda/miniforge discovery, and the
`cpsam_foundation310` Conda environment. The jobs also try to load
`CUDA_MODULE=cuda/12.1`; this is non-fatal because the Imperial PyTorch guide
configures CUDA/cuDNN primarily inside the Conda environment and module names
can vary. Each job requests one GPU, 16 CPU cores, and 48 GB RAM.

Submit all three with PBS dependencies from the repository root:

```bash
bash jobs/submit_cpsam_student_fpn_three_stage.pbs.sh
```

Override defaults if needed:

```bash
REPO_DIR="$HOME/MultiCellPose" \
TRAIN_ROOT_DIRS="$HOME/FoundationTrain" \
OUTPUT_DIR="$HOME/FoundationTrain/distilled_cpsam_encoder_fpn_three_stage" \
CONDA_ENV="cpsam_foundation310" \
CUDA_MODULE="cuda/12.1" \
STAGE3_FLOW_CACHE_DIR="${EPHEMERAL:-$HOME/FoundationTrain/distilled_cpsam_encoder_fpn_three_stage}/cpsam_fpn_supervised_flow_cache" \
bash jobs/submit_cpsam_student_fpn_three_stage.pbs.sh
```

Create the CUDA 12.1 environment using the Imperial guide pattern, adapted from
CUDA 11.8 to CUDA 12.1:

```bash
if [ -x "$HOME/miniconda3/bin/conda" ]; then
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
elif [ -x "$HOME/miniforge3/bin/conda" ]; then
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
else
    echo "ERROR: conda not found"
    exit 2
fi

conda create -n cpsam_foundation310 -c conda-forge cudatoolkit=12.1 python=3.10 -y
conda activate cpsam_foundation310
conda install -c "nvidia/label/cuda-12.1.0" cuda-nvcc -y
python -m pip install nvidia-cudnn-cu12

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" <<'EOF'
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
EOF
source "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify the CUDA 12.1 PyTorch environment from a GPU job:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

The stage 2 job initializes from:

```text
<OUTPUT_DIR>/stage1_feature/cpsam_encoder_student_best.pt
```

The stage 3 job initializes from:

```text
<OUTPUT_DIR>/stage2_output/cpsam_encoder_student_best.pt
```

The stage 3 job passes `--stage3-flow-device cuda`, so missing supervised flow
targets are precomputed on the GPU. By default the cache is written to
`${EPHEMERAL:-$OUTPUT_DIR}/cpsam_fpn_supervised_flow_cache`, following the
validated foundation job's use of ephemeral storage for large flow caches.

For a single PBS submission that runs all three stages in one Python process:

```bash
qsub jobs/train_cpsam_student_fpn_three_stage.pbs
```

That job passes the stage counts directly to
`tools/train_cpsam_student_fpn_three_stage.py`:

```text
--stage1-epochs 300 --stage2-epochs 200 --stage3-epochs 300
```

Set the `#PBS -l walltime=...` line in the single PBS file according to whether
you are running a short smoke test or the full 300/200/300 epoch schedule. If
your queue does not permit the required walltime for the full schedule, use the
staged jobs instead.
