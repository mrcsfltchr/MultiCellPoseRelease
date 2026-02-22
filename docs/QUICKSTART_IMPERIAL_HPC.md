# Quick Start: Foundation Training on Imperial HPC (PBS)

Simplified step-by-step guide for training CPSAM foundation models on Imperial's PBS cluster.

---

## Before You Start

**On your local machine:**

1. ✅ Organize your dataset:
   ```
   my_dataset/
     ├─ image001.tif          ├─ image001_masks.tif
     ├─ stack002.nd2          ├─ stack002_seg.npy
     ├─ series003.lif         ├─ series003_masks.png
     └─ ...
   ```

2. ✅ Transfer to cluster:
   ```bash
   # Replace <username> with your Imperial username
   rsync -avP my_dataset/ <username>@login.hpc.ic.ac.uk:~/rds/foundation_training/raw/
   ```

---

## Step-by-Step Setup (On HPC Cluster)

### 1. Login and Verify Data

```bash
# Login to cluster
ssh <username>@login.hpc.ic.ac.uk

# Check data arrived
cd ~/rds/foundation_training/raw
ls -lh | head -20

# Count images and labels
find . -type f \( -name "*.tif" -o -name "*.nd2" -o -name "*.lif" \) ! -name "*_masks.*" ! -name "*_seg.*" | wc -l
find . -type f \( -name "*_masks.*" -o -name "*_seg.npy" \) | wc -l
# These numbers should match!
```

### 2. Load Modules and Setup Environment

```bash
# Load required modules
module load python/3.9
module load cuda/12.1

# Clone repository
cd ~
git clone https://github.com/YOUR_USERNAME/MultiCellPose-public.git
cd MultiCellPose-public

# Create virtual environment
python3 -m venv venv_foundation
source venv_foundation/bin/activate

# Install packages
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. Download SAM Checkpoint

```bash
# Download original SAM encoder
mkdir -p ~/.cellpose/models
wget -O ~/.cellpose/models/sam_vit_l_0b3195.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# Verify (should be ~1.2GB)
ls -lh ~/.cellpose/models/sam_vit_l_0b3195.pth
```

### 4. Prepare Training Directory

```bash
# Create directories
cd ~/rds/foundation_training
mkdir -p train logs models

# Copy/link data to training dir
# Option 1 (faster, uses less space):
ln -s $(pwd)/raw/*.tif train/
ln -s $(pwd)/raw/*_masks.* train/
ln -s $(pwd)/raw/*_seg.npy train/

# Option 2 (safer, uses more space):
# cp raw/* train/

# Verify
ls train/ | wc -l
# Should be ~2× your image count (images + labels)
```

### 5. Create PBS Job Script

```bash
cd ~/MultiCellPose-public
mkdir -p jobs

# Create 4-GPU training job
cat > jobs/train_foundation.pbs << 'EOF'
#!/bin/bash
#PBS -N cpsam_foundation
#PBS -l select=1:ncpus=16:ngpus=4:mem=96gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o ~/rds/foundation_training/logs/train_${PBS_JOBID}.log

# Load modules
module load python/3.9
module load cuda/12.1

# Activate environment
cd $PBS_O_WORKDIR
source venv_foundation/bin/activate

# Configuration
TRAIN_DIR=~/rds/foundation_training/train
OUTPUT_DIR=~/rds/foundation_training/models
SAM_CHECKPOINT=~/.cellpose/models/sam_vit_l_0b3195.pth

# Log info
echo "Job: $PBS_JOBID | Node: $(hostname) | Start: $(date)"
nvidia-smi
echo "========================================="

# Run training (4 GPUs with DDP)
torchrun --nproc_per_node=4 scripts/train_cpsam_foundation_multigpu.py \
    --train-dir $TRAIN_DIR \
    --foundation-training \
    --sam-checkpoint $SAM_CHECKPOINT \
    --multi-gpu ddp \
    --batch-size 40 \
    --epochs 300 \
    --unfreeze-blocks 9 \
    --learning-rate 5e-5 \
    --weight-decay 0.1 \
    --bsize 256 \
    --test-ratio 0.1 \
    --save-path $OUTPUT_DIR \
    --model-name cpsam_foundation_$(date +%Y%m%d) \
    --verbose

echo "========================================="
echo "Finished: $(date)"
EOF
```

### 6. Submit Job

```bash
# Submit to queue
cd ~/MultiCellPose-public
qsub jobs/train_foundation.pbs

# Check status
qstat -u $USER

# Monitor output
tail -f ~/rds/foundation_training/logs/train_*.log
```

---

## Monitoring Your Job

### Check Job Status

```bash
# List your jobs
qstat -u $USER

# Detailed job info
qstat -f <job_id>

# Why isn't my job running?
checkjob <job_id>
```

### Monitor Training Progress

```bash
# Watch log file
tail -f ~/rds/foundation_training/logs/train_*.log

# Expected output:
# [Rank 0] ... - INFO - epoch 1/300 train_loss=0.4521 test_loss=0.4123
# [Rank 0] ... - INFO - epoch 2/300 train_loss=0.4312 test_loss=0.4001
```

### Cancel Job (if needed)

```bash
# Cancel job
qdel <job_id>
```

---

## After Training Completes

### 1. Check Model Files

```bash
# List trained models
ls -lh ~/rds/foundation_training/models/

# You should see:
# cpsam_foundation_YYYYMMDD
```

### 2. Download Model to Local Machine

```bash
# On your local machine
rsync -avP <username>@login.hpc.ic.ac.uk:~/rds/foundation_training/models/ \
  ./trained_models/
```

### 3. Test Model

```bash
# On local machine or cluster
python << 'EOF'
from cellpose.models import CellposeModel
from cellpose import io

# Load trained model
model = CellposeModel(
    pretrained_model="trained_models/cpsam_foundation_YYYYMMDD",
    gpu=True
)

# Test on image
img = io.imread("test_image.tif")
masks, flows, styles = model.eval(img, diameter=30)
print(f"Found {masks.max()} cells")
EOF
```

---

## Common Issues & Solutions

### "qsub: submit error (Unauthorized Request)"
**Fix:** Your account may not have access to GPU queue. Contact RCS support.

### "CUDA out of memory"
**Fix:** Reduce batch size in job script:
```bash
--batch-size 20  # Instead of 40 (now 5 per GPU)
```

### "Module not found: python/3.9"
**Fix:** Check available modules and adjust:
```bash
module avail python
module load python/3.10  # Use whatever is available
```

### "Job pending in queue forever"
**Fix:** Check queue limits and reduce resources:
```bash
# Reduce GPUs:
#PBS -l select=1:ncpus=8:ngpus=2:mem=48gb

# Reduce walltime:
#PBS -l walltime=24:00:00

# Then in script, change:
torchrun --nproc_per_node=2 ...  # Match GPU count
--batch-size 20  # Adjust for fewer GPUs
```

---

## Recommended First Run (Test)

Before launching a full 300-epoch run, test with a smaller job:

```bash
# Edit jobs/train_foundation.pbs, change:
#PBS -l walltime=02:00:00  # 2 hours instead of 48

# And in torchrun command:
--epochs 10  # Just 10 epochs to verify everything works
--batch-size 20  # Smaller batch for safety

# Then submit:
qsub jobs/train_foundation.pbs
```

If this completes successfully, launch the full run!

---

## Estimated Training Times

| Images | GPUs | Epochs | Walltime Needed |
|--------|------|--------|-----------------|
| 500    | 4    | 300    | 12 hours        |
| 1000   | 4    | 300    | 24 hours        |
| 2000   | 4    | 300    | 48 hours        |

---

## Need Help?

- **Imperial RCS Support**: https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/support/
- **Check Documentation**: `~/MultiCellPose-public/docs/HPC_FOUNDATION_TRAINING_GUIDE.md`
- **Quick Commands**: `~/MultiCellPose-public/docs/HPC_FOUNDATION_TRAINING_GUIDE.md` → Appendix B
