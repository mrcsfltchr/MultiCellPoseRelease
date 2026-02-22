# Foundation Model Training on Imperial HPC Cluster

Complete guide for training CPSAM foundation models on Imperial College London's HPC cluster with your custom microscopy dataset.

## Prerequisites

- Access to Imperial HPC cluster (login node)
- Custom dataset with mixed formats (.tif, .nd2, .lif multistack files)
- Labels as `_masks.tif`, `_masks.png`, or `_seg.npy`
- ~50GB+ disk space for dataset + features
- Basic familiarity with Linux command line

---

## Part 1: Dataset Preparation

### Step 1.1: Organize Your Local Dataset

Before uploading to the cluster, organize your dataset locally:

```bash
# On your local machine, create organized structure
mkdir -p foundation_training_data/raw
cd foundation_training_data/raw

# Copy all your image files and labels here
# Expected structure:
raw/
  ├─ experiment1_image001.tif
  ├─ experiment1_image001_masks.tif
  ├─ experiment2_stack.nd2
  ├─ experiment2_stack_seg.npy
  ├─ experiment3_series.lif
  ├─ experiment3_series_masks.png
  └─ ...
```

### Step 1.2: Create Dataset Manifest

Create a simple inventory of your data:

```bash
# On your local machine
cd foundation_training_data

# List all images (excluding masks/labels)
find raw -type f \( -name "*.tif" -o -name "*.nd2" -o -name "*.lif" \) \
  ! -name "*_masks.*" ! -name "*_seg.*" > image_list.txt

# Count images
wc -l image_list.txt
# Example output: 1234 image_list.txt

# List all label files
find raw -type f \( -name "*_masks.tif" -o -name "*_masks.png" -o -name "*_seg.npy" \) > label_list.txt

# Verify counts match
wc -l label_list.txt
```

**Important:** Ensure every image has a corresponding label file. If counts don't match, identify missing labels:

```bash
# Check for images without labels
python3 << 'EOF'
import os
from pathlib import Path

# Read image list
with open('image_list.txt') as f:
    images = [line.strip() for line in f]

# Check each image has a label
missing_labels = []
for img in images:
    img_path = Path(img)
    stem = img_path.stem
    parent = img_path.parent

    # Check for any label variant
    has_label = any([
        (parent / f"{stem}_masks.tif").exists(),
        (parent / f"{stem}_masks.png").exists(),
        (parent / f"{stem}_seg.npy").exists(),
    ])

    if not has_label:
        missing_labels.append(img)

if missing_labels:
    print(f"WARNING: {len(missing_labels)} images missing labels:")
    for m in missing_labels[:10]:
        print(f"  {m}")
else:
    print("✓ All images have labels!")
EOF
```

### Step 1.3: Transfer Data to HPC Cluster

```bash
# On your local machine
# Replace <username> and <cluster-address> with your Imperial credentials

# Transfer dataset to cluster
rsync -avP --info=progress2 foundation_training_data/ \
  <username>@<cluster-address>:~/rds/training_data/foundation_cpsam/

# Example for Imperial (adjust based on actual cluster address):
# rsync -avP foundation_training_data/ username@login.hpc.ic.ac.uk:~/rds/training_data/foundation_cpsam/
```

**Note:** `~/rds/` is typically the Research Data Store location on Imperial's HPC. Check with `echo $RDS` on the cluster to confirm.

---

## Part 2: HPC Environment Setup

### Step 2.1: Login to HPC Cluster

```bash
# On your local machine
ssh <username>@<cluster-address>

# Example:
# ssh username@login.hpc.ic.ac.uk
```

### Step 2.2: Check Data Transfer

```bash
# On HPC login node
cd ~/rds/training_data/foundation_cpsam/raw
ls -lh | head -20

# Verify file counts
find . -name "*.tif" ! -name "*_masks.*" | wc -l
find . -name "*_masks.*" -o -name "*_seg.npy" | wc -l
```

### Step 2.3: Load Required Modules

```bash
# On HPC login node
# Check available Python/CUDA modules
module avail python
module avail cuda

# Load modules (adjust versions based on availability)
module load python/3.9
module load cuda/12.1
module load cudnn/8.9

# Verify
python3 --version
nvcc --version
```

**If modules aren't available:** Create a conda environment (see Appendix A).

### Step 2.4: Install MultiCellPose

```bash
# On HPC login node
cd ~
git clone https://github.com/YOUR_USERNAME/MultiCellPose-public.git
cd MultiCellPose-public

# Create virtual environment
python3 -m venv venv_foundation
source venv_foundation/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install fastapi uvicorn  # For HTTP gateway (optional)

# Verify installation
python -c "import cellpose; print(cellpose.__version__)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2.5: Download SAM Checkpoint

```bash
# On HPC login node
mkdir -p ~/.cellpose/models
cd ~/.cellpose/models

# Download original SAM encoder checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# Verify download
ls -lh sam_vit_l_0b3195.pth
# Expected: ~1.2GB file
```

---

## Part 3: Prepare Training Dataset

### Step 3.1: Create Training Directory Structure

```bash
# On HPC login node
cd ~/rds/training_data/foundation_cpsam
mkdir -p train
mkdir -p logs
mkdir -p models
```

### Step 3.2: Convert Dataset to Standard Format (if needed)

For multistack files (.nd2, .lif), you may need to extract frames first:

```bash
# On HPC login node
cd ~/MultiCellPose-public

# Create extraction script
cat > scripts/extract_multistack.py << 'EOF'
#!/usr/bin/env python3
"""
Extract frames from multistack microscopy files (.nd2, .lif) for training.
"""
import argparse
from pathlib import Path
import numpy as np
from cellpose import io

def extract_frames(input_file, output_dir, max_frames=None):
    """Extract frames from multistack file."""
    print(f"Processing {input_file.name}...")

    # Read multistack file
    img = io.imread(str(input_file))

    # Handle different dimensionalities
    if img.ndim == 2:
        # Single image
        frames = [img]
    elif img.ndim == 3:
        # Could be (T, H, W) or (H, W, C)
        if img.shape[-1] <= 4:  # Likely (H, W, C)
            frames = [img]
        else:  # Likely (T, H, W)
            frames = [img[i] for i in range(img.shape[0])]
    elif img.ndim == 4:
        # Likely (T, H, W, C)
        frames = [img[i] for i in range(img.shape[0])]
    else:
        print(f"  WARNING: Unexpected shape {img.shape}, skipping")
        return []

    # Limit frames if requested
    if max_frames:
        frames = frames[:max_frames]

    # Save frames
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem
    saved_paths = []

    for i, frame in enumerate(frames):
        output_path = output_dir / f"{stem}_frame{i:04d}.tif"
        io.imsave(str(output_path), frame)
        saved_paths.append(output_path)

    print(f"  Extracted {len(frames)} frames")
    return saved_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory with .nd2/.lif files")
    parser.add_argument("output_dir", help="Output directory for frames")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Max frames per stack (default: all)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Find multistack files
    multistack_files = list(input_dir.glob("*.nd2")) + list(input_dir.glob("*.lif"))
    print(f"Found {len(multistack_files)} multistack files")

    for mfile in multistack_files:
        extract_frames(mfile, output_dir, args.max_frames)

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/extract_multistack.py

# Extract frames (if you have multistack files)
python scripts/extract_multistack.py \
    ~/rds/training_data/foundation_cpsam/raw \
    ~/rds/training_data/foundation_cpsam/extracted \
    --max-frames 100
```

### Step 3.3: Symlink or Copy to Training Directory

```bash
# On HPC login node
cd ~/rds/training_data/foundation_cpsam

# Option 1: Symlink (saves space, data stays in raw/)
ln -s $(pwd)/raw/*.tif train/
ln -s $(pwd)/raw/*_masks.* train/
ln -s $(pwd)/raw/*_seg.npy train/

# Option 2: Copy (safer for training, uses more space)
cp raw/*.tif train/
cp raw/*_masks.* train/
cp raw/*_seg.npy train/

# If you extracted multistack frames:
cp extracted/*.tif train/

# Verify
ls train/ | wc -l
# Should see ~2x the number of images (images + labels)
```

---

## Part 4: Multi-GPU Training Job Submission

### Step 4.1: Create Job Submission Script (PBS)

If your cluster uses PBS (most common for Imperial):

```bash
# On HPC login node
cd ~/MultiCellPose-public

cat > jobs/train_foundation_4gpu.pbs << 'EOF'
#!/bin/bash
#PBS -N cpsam_foundation
#PBS -l select=1:ncpus=16:ngpus=4:mem=96gb:gpu_type=A100
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o ~/rds/training_data/foundation_cpsam/logs/train_${PBS_JOBID}.log

# Load modules
module load python/3.9
module load cuda/12.1
module load cudnn/8.9

# Activate environment
cd $PBS_O_WORKDIR
source venv_foundation/bin/activate

# Set paths
TRAIN_DIR=~/rds/training_data/foundation_cpsam/train
OUTPUT_DIR=~/rds/training_data/foundation_cpsam/models
SAM_CHECKPOINT=~/.cellpose/models/sam_vit_l_0b3195.pth

# Training configuration
BATCH_SIZE=40  # Total across 4 GPUs = 10 per GPU
EPOCHS=300
UNFREEZE_BLOCKS=9
LEARNING_RATE=5e-5

# Log environment info
echo "========================================="
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================="
nvidia-smi
echo "========================================="

# Run multi-GPU training with DDP (most efficient)
torchrun --nproc_per_node=4 scripts/train_cpsam_foundation_multigpu.py \
    --train-dir $TRAIN_DIR \
    --foundation-training \
    --sam-checkpoint $SAM_CHECKPOINT \
    --multi-gpu ddp \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --unfreeze-blocks $UNFREEZE_BLOCKS \
    --learning-rate $LEARNING_RATE \
    --weight-decay 0.1 \
    --bsize 256 \
    --test-ratio 0.1 \
    --save-path $OUTPUT_DIR \
    --model-name cpsam_foundation_imperial_v1 \
    --verbose

echo "========================================="
echo "End time: $(date)"
echo "========================================="
EOF

chmod +x jobs/train_foundation_4gpu.pbs
```

### Step 4.2: Create Job Submission Script (SLURM)

If your cluster uses SLURM instead:

```bash
# On HPC login node
cat > jobs/train_foundation_4gpu.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=cpsam_foundation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=~/rds/training_data/foundation_cpsam/logs/train_%j.log

# Load modules
module load python/3.9
module load cuda/12.1

# Activate environment
cd $SLURM_SUBMIT_DIR
source venv_foundation/bin/activate

# Set paths
TRAIN_DIR=~/rds/training_data/foundation_cpsam/train
OUTPUT_DIR=~/rds/training_data/foundation_cpsam/models
SAM_CHECKPOINT=~/.cellpose/models/sam_vit_l_0b3195.pth

# Log environment
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi

# Run training
srun torchrun --nproc_per_node=4 scripts/train_cpsam_foundation_multigpu.py \
    --train-dir $TRAIN_DIR \
    --foundation-training \
    --sam-checkpoint $SAM_CHECKPOINT \
    --multi-gpu ddp \
    --batch-size 40 \
    --epochs 300 \
    --unfreeze-blocks 9 \
    --learning-rate 5e-5 \
    --save-path $OUTPUT_DIR \
    --model-name cpsam_foundation_imperial_v1 \
    --verbose
EOF

chmod +x jobs/train_foundation_4gpu.slurm
```

### Step 4.3: Submit Training Job

```bash
# On HPC login node

# For PBS:
cd ~/MultiCellPose-public
qsub jobs/train_foundation_4gpu.pbs

# For SLURM:
sbatch jobs/train_foundation_4gpu.slurm

# Check job status (PBS):
qstat -u $USER

# Check job status (SLURM):
squeue -u $USER

# Monitor job output:
tail -f ~/rds/training_data/foundation_cpsam/logs/train_*.log
```

---

## Part 5: Monitoring Training

### Step 5.1: Real-time Log Monitoring

```bash
# On HPC login node (while job is running)
tail -f ~/rds/training_data/foundation_cpsam/logs/train_*.log

# Expected output:
# [Rank 0] 2024-02-18 10:00:00 - INFO - Initialized DDP: rank=0, world_size=4
# [Rank 0] 2024-02-18 10:00:05 - INFO - Multi-GPU training enabled: 4 GPUs
# [Rank 0] 2024-02-18 10:01:00 - INFO - epoch 1/300 train_loss=0.4521
```

### Step 5.2: GPU Utilization (if you can SSH to compute node)

```bash
# If allowed, SSH to compute node
# Get node name from qstat/squeue
ssh <compute-node-name>
watch -n 1 nvidia-smi
```

### Step 5.3: Check Model Checkpoints

```bash
# On HPC login node
ls -lh ~/rds/training_data/foundation_cpsam/models/

# You should see:
# cpsam_foundation_imperial_v1  (final model)
# Or intermediate checkpoints if saving is enabled
```

---

## Part 6: After Training

### Step 6.1: Download Trained Model

```bash
# On your local machine
rsync -avP <username>@<cluster-address>:~/rds/training_data/foundation_cpsam/models/ \
  ./trained_models/
```

### Step 6.2: Test Trained Model

```bash
# On your local machine or on cluster
python << 'EOF'
from cellpose.models import CellposeModel
from cellpose import io
import numpy as np

# Load your trained foundation model
model = CellposeModel(
    pretrained_model="trained_models/cpsam_foundation_imperial_v1",
    gpu=True
)

# Test on a sample image
test_img = io.imread("path/to/test_image.tif")
masks, flows, styles = model.eval(test_img, diameter=30)

print(f"Detected {masks.max()} cells")
EOF
```

---

## Troubleshooting

### Problem: Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce batch size or crop size:

```bash
# Edit job script, reduce:
--batch-size 20  # Instead of 40 (5 per GPU instead of 10)
--bsize 224      # Instead of 256
```

### Problem: Module Not Found

```
ModuleNotFoundError: No module named 'cellpose'
```

**Solution:** Ensure virtual environment is activated:

```bash
source ~/MultiCellPose-public/venv_foundation/bin/activate
pip list | grep cellpose
```

### Problem: Job Doesn't Start

**PBS:**
```bash
qstat -f <job_id>  # Check detailed status
checkjob <job_id>  # Check why job isn't running
```

**SLURM:**
```bash
scontrol show job <job_id>
```

**Common reasons:**
- Requested resources not available (reduce GPU count or walltime)
- Wrong partition/queue name
- Insufficient disk quota

### Problem: SAM Checkpoint Not Found

```
FileNotFoundError: SAM checkpoint not found
```

**Solution:** Re-download:

```bash
cd ~/.cellpose/models
rm sam_vit_l_0b3195.pth  # Remove corrupted download
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
md5sum sam_vit_l_0b3195.pth
# Expected: 0b3195507c641ddb6910d2bb5adee89c
```

---

## Appendix A: Alternative Setup with Conda

If module system doesn't work, use Conda:

```bash
# On HPC login node
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source ~/miniconda3/bin/activate

# Create environment
conda create -n cpsam_foundation python=3.9 -y
conda activate cpsam_foundation

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install cellpose
cd ~/MultiCellPose-public
pip install -e .
```

Then modify job script to use:
```bash
source ~/miniconda3/bin/activate
conda activate cpsam_foundation
```

---

## Appendix B: Quick Reference Commands

```bash
# Check job queue (PBS)
qstat -u $USER

# Cancel job (PBS)
qdel <job_id>

# Check job queue (SLURM)
squeue -u $USER

# Cancel job (SLURM)
scancel <job_id>

# Monitor GPU on compute node
ssh <node-name> nvidia-smi

# Check disk usage
du -sh ~/rds/training_data/foundation_cpsam/*

# Monitor training log
tail -f ~/rds/training_data/foundation_cpsam/logs/train_*.log
```

---

## Expected Training Time

Based on typical HPC GPU performance:

| Dataset Size | GPUs | Batch Size | Epochs | Expected Time |
|--------------|------|------------|--------|---------------|
| 500 images   | 4    | 40         | 300    | ~12 hours     |
| 1000 images  | 4    | 40         | 300    | ~24 hours     |
| 2000 images  | 4    | 40         | 300    | ~48 hours     |

**Recommendation:** Start with 50-100 epochs on a subset to verify everything works, then launch full 300-epoch run.

---

## Contact for Help

- **Imperial RCS Support**: https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/support/
- **Imperial HPC Documentation**: https://icl-rcs-user-guide.readthedocs.io/
- **MultiCellPose Issues**: File issues at your GitHub repo
