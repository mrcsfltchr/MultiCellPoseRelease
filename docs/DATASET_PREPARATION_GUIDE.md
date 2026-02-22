# Dataset Preparation Guide for Foundation Training

Quick guide for preparing your foundation training dataset from `X:\home\FoundationTrain`.

## Dataset Overview

Your current dataset structure:
```
X:/home/FoundationTrain/
├── bact_fluor/              # Fluorescent bacteria (143 train images)
│   ├── train_sorted/
│   └── test_sorted/
├── bact_phase/              # Phase contrast bacteria (249 train images)
│   ├── train_sorted/
│   └── test_sorted/
├── worm/                    # Worm dataset (60 train images)
│   ├── train/
│   └── test/
├── worm_high_res/           # High-res worm (48 train images)
│   ├── train/
│   └── test/
├── tissuenet_v1.1_train.npz   # TissueNet training (~2.7GB, 2601 images)
├── tissuenet_v1.1_val.npz     # TissueNet validation (~1.2GB, 1249 images)
└── tissuenet_v1.1_test.npz    # TissueNet test (~371MB)
```

**Total without TissueNet:** ~500 images
**Total with TissueNet:** ~4,000+ images

---

## Quick Start

### Option 1: Prepare Dataset WITHOUT TissueNet (Fast)

If you want to start quickly with the smaller datasets:

```bash
python scripts/prepare_foundation_dataset.py \
    --input-dir X:/home/FoundationTrain \
    --output-dir data/foundation_cpsam \
    --datasets bact_fluor bact_phase worm worm_high_res \
    --create-train-weights \
    --verbose
```

**Time:** ~30 seconds
**Output:** ~500 images ready for training
**Disk space:** ~2-3 GB

### Option 2: Prepare Dataset WITH TissueNet (Recommended)

For the full foundation training experience:

```bash
python scripts/prepare_foundation_dataset.py \
    --input-dir X:/home/FoundationTrain \
    --output-dir data/foundation_cpsam \
    --extract-tissuenet \
    --tissuenet-max-train 2000 \
    --create-train-weights \
    --verbose
```

**Time:** ~5-10 minutes (extracting from .npz)
**Output:** ~2,500 images
**Disk space:** ~15-20 GB

### Option 3: Use Symlinks to Save Space

If disk space is limited:

```bash
python scripts/prepare_foundation_dataset.py \
    --input-dir X:/home/FoundationTrain \
    --output-dir data/foundation_cpsam \
    --create-symlinks \
    --create-train-weights \
    --verbose
```

**Disk space:** <1 GB (just symlinks, not copies)
**Note:** Original files must remain in place

---

## Understanding the Sampling Weights

The script creates `sampling_weights.json` which implements the cellpose-SAM sampling strategy:

| Dataset | Your Count | CPSAM Sampling % | Rationale |
|---------|-----------|------------------|-----------|
| bact_fluor | 143 | 1% | Specialized, don't over-sample |
| bact_phase | 249 | 2% | Specialized, moderate sampling |
| worm | 60 | 0% | Custom (not in CPSAM paper) |
| worm_high_res | 48 | 0% | Custom (not in CPSAM paper) |
| tissuenet | 2,601 | 8% | Diverse tissue types |

**Key insight from paper:** "We upweighted images in the cyto2 and nuclei training sets because they contained the most variability across images."

Since you don't have cyto2/nuclei datasets yet, TissueNet will be your most diverse dataset and should receive higher sampling weight.

### Adjusting Sampling Weights

Edit `scripts/prepare_foundation_dataset.py` if you want to change the weights:

```python
# Around line 33
DATASET_SAMPLING_WEIGHTS = {
    "bact_fluor": 0.01,     # 1% - fluorescent bacteria
    "bact_phase": 0.02,     # 2% - phase bacteria
    "worm": 0.05,           # 5% - increase if worms are important
    "worm_high_res": 0.02,  # 2% - high-res worms
    "tissuenet": 0.90,      # 90% - main diverse dataset
}
```

---

## Output Structure

After running the script, you'll have:

```
data/foundation_cpsam/
├── train/                           # Combined training set
│   ├── bact_fluor__A22_s_flex_a22_t12xy5c1_tile10_cyto.tif
│   ├── bact_fluor__A22_s_flex_a22_t12xy5c1_tile10_cyto_masks.tif
│   ├── bact_phase__PAK_m001xy1_frame_1.tif
│   ├── bact_phase__PAK_m001xy1_frame_1_masks.tif
│   ├── worm__001.tif
│   ├── worm__001_masks.tif
│   ├── tissuenet__train_00042.tif
│   ├── tissuenet__train_00042_seg.npy
│   └── ...
├── test/                            # Combined test set
│   └── ...
└── sampling_weights.json            # Dataset sampling configuration
```

**File naming convention:**
```
{dataset}__{subset}_{original_name}.{ext}
```

This makes it easy to identify which dataset each image came from.

---

## Verifying the Prepared Dataset

### 1. Check File Counts

```bash
# Count images in training set
find data/foundation_cpsam/train -name "*.tif" -o -name "*.png" | \
  grep -v "_masks" | grep -v "_seg" | wc -l

# Count labels
find data/foundation_cpsam/train -name "*_masks.*" -o -name "*_seg.npy" | wc -l

# These should match!
```

### 2. Verify Pairing

```python
python << 'EOF'
from pathlib import Path

train_dir = Path("data/foundation_cpsam/train")

# Find all images (excluding masks)
images = []
for ext in [".tif", ".png"]:
    images.extend([
        f for f in train_dir.glob(f"*{ext}")
        if "_masks" not in f.name and "_seg" not in f.name
    ])

# Check each has a mask
missing = []
for img in images:
    has_mask = any([
        (train_dir / f"{img.stem}_masks.tif").exists(),
        (train_dir / f"{img.stem}_masks.png").exists(),
        (train_dir / f"{img.stem}_seg.npy").exists(),
    ])
    if not has_mask:
        missing.append(img.name)

if missing:
    print(f"WARNING: {len(missing)} images missing masks:")
    for m in missing[:10]:
        print(f"  {m}")
else:
    print(f"✓ All {len(images)} images have masks!")
EOF
```

### 3. Inspect Sampling Weights

```bash
cat data/foundation_cpsam/sampling_weights.json
```

Expected output:
```json
{
  "total_images": 2500,
  "dataset_counts": {
    "bact_fluor": 143,
    "bact_phase": 249,
    "worm": 60,
    "worm_high_res": 48,
    "tissuenet": 2000
  },
  "dataset_sampling_probs": {
    "bact_fluor": 0.01,
    "bact_phase": 0.02,
    ...
  },
  "per_image_weights": {
    "bact_fluor": 0.1748,
    "tissuenet": 0.1000,
    ...
  }
}
```

**per_image_weights** tells the training script how much to weight each dataset during sampling.

---

## Adding More Datasets

If you acquire additional datasets (e.g., LiveCell, Cellpose cyto2, nuclei):

1. **Add to input directory:**
   ```
   X:/home/FoundationTrain/livecell/
     ├── train/
     │   ├── image_001.png
     │   ├── image_001_masks.png
     │   └── ...
     └── test/
   ```

2. **Update the script:**
   Edit `scripts/prepare_foundation_dataset.py`:
   ```python
   # Add to dataset choices (line ~265)
   choices=["all", "bact_fluor", "bact_phase", "worm", "livecell", ...]

   # Add to sampling weights (line ~33)
   DATASET_SAMPLING_WEIGHTS = {
       ...
       "livecell": 0.05,
   }

   # Add processing section (around line 520)
   if "livecell" in all_datasets:
       logger.info("\nProcessing livecell...")
       pairs = find_image_mask_pairs(
           input_dir / "livecell" / "train",
           "livecell",
           recursive=True
       )
       copied = copy_and_rename(pairs, train_dir, args.create_symlinks)
       dataset_counts["livecell"] = copied
   ```

3. **Re-run preparation:**
   ```bash
   python scripts/prepare_foundation_dataset.py \
       --input-dir X:/home/FoundationTrain \
       --output-dir data/foundation_cpsam \
       --datasets all \
       --create-train-weights
   ```

---

## Troubleshooting

### "No mask found for X.tif"

**Cause:** Image and mask have different naming conventions.

**Solution:** Check the actual filenames:
```bash
ls X:/home/FoundationTrain/bact_fluor/train_sorted/A22/ | grep tile10
```

If masks are named differently (e.g., `_labels.tif` instead of `_masks.tif`), update the script:
```python
# Around line 210
mask_candidates = [
    img_path.parent / f"{img_path.stem}_masks.tif",
    img_path.parent / f"{img_path.stem}_labels.tif",  # Add this
    img_path.parent / f"{img_path.stem}_seg.npy",
]
```

### "Could not identify image/mask keys in tissuenet"

**Cause:** .npz file has unexpected structure.

**Solution:** Inspect manually:
```python
import numpy as np
data = np.load("X:/home/FoundationTrain/tissuenet_v1.1_train.npz")
print("Keys:", list(data.keys()))
for key in data.keys():
    print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
```

Then update the key detection in the script (around line 88).

### "Out of disk space"

**Solution 1:** Use symlinks:
```bash
python scripts/prepare_foundation_dataset.py ... --create-symlinks
```

**Solution 2:** Limit TissueNet:
```bash
python scripts/prepare_foundation_dataset.py ... --tissuenet-max-train 500
```

**Solution 3:** Skip TissueNet entirely:
```bash
python scripts/prepare_foundation_dataset.py ... --datasets bact_fluor bact_phase worm
```

---

## Next Steps

After dataset preparation:

1. **Verify dataset:**
   ```bash
   python scripts/load_cpsam_features.py data/foundation_cpsam/train --verify
   ```

2. **Train foundation model:**
   ```bash
   python scripts/train_cpsam_foundation.py \
       --train-dir data/foundation_cpsam/train \
       --foundation-training \
       --sam-checkpoint ~/.cellpose/models/sam_vit_l_0b3195.pth \
       --epochs 300 \
       --batch-size 40 \
       --model-name foundation_mixed_v1
   ```

3. **For HPC cluster training:**
   See `docs/QUICKSTART_IMPERIAL_HPC.md`
