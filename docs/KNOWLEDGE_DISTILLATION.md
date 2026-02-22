# Knowledge Distillation for CPSAM

This document explains how to prepare CPSAM encoder features for knowledge distillation training, where a smaller student model learns from the pre-trained CPSAM encoder.

## Overview

Knowledge distillation allows you to train smaller, faster models by transferring knowledge from a large teacher model (CPSAM) to a compact student model. The process involves:

1. **Feature Extraction**: Run images through the CPSAM encoder and save the 256-channel intermediate features
2. **Student Training**: Train a smaller model to reproduce both the CPSAM features and the segmentation outputs

This workflow focuses on Step 1 - extracting and saving CPSAM encoder features efficiently.

## Architecture: Where Features Are Extracted

```
CPSAM Model
  ├─ Encoder
  │   ├─ Patch Embedding (8×8 patches)
  │   ├─ Positional Embeddings
  │   ├─ Transformer Blocks [32 blocks]
  │   └─ Neck → 256-channel features  ← EXTRACTION POINT
  │
  └─ Segmentation Head (not used during extraction)
      ├─ Flow Y prediction
      ├─ Flow X prediction
      └─ Cell probability prediction
```

**Key point**: We extract features *after* the encoder neck but *before* the segmentation head. These 256-channel features capture rich visual representations that can guide student model training.

## Step 1: Extract CPSAM Encoder Features

### Prerequisites

- CPSAM model installed and working
- Image dataset prepared (any format supported by cellpose: `.png`, `.jpg`, `.tif`, `.bmp`, etc.)
- GPU recommended for faster extraction

### Command-Line Usage

```bash
python scripts/extract_cpsam_features.py \
    --image-dir data/train_images \
    --output-dir data/cpsam_features \
    --model cpsam \
    --gpu \
    --batch-size 8 \
    --bsize 256
```

**Arguments:**
- `--image-dir`: Directory containing images to process
- `--output-dir`: Where to save extracted features (will be created if needed)
- `--model`: CPSAM model to use (default: `cpsam`, can be path to custom model)
- `--gpu`: Use GPU acceleration (recommended)
- `--batch-size`: Number of tiles to process at once (default: 8, reduce if OOM)
- `--bsize`: Tile size for processing (default: 256)
- `--verbose`: Enable detailed logging

### Output Structure

After extraction, you'll have:

```
data/cpsam_features/
  ├─ manifest.json                # Metadata for all extracted features
  ├─ features_image_001.npz       # Compressed features for image_001
  ├─ features_image_002.npz
  └─ ...
```

**manifest.json** contains:
```json
{
  "model": "cpsam",
  "feature_dim": 256,
  "num_images": 1234,
  "features": [
    {
      "image_id": "image_001",
      "image_path": "data/train_images/image_001.png",
      "original_shape": [512, 512],
      "feature_shape": [64, 64],
      "feature_file": "features_image_001.npz"
    },
    ...
  ]
}
```

**Each .npz file** contains:
- `features`: (256, H, W) numpy array - the encoder features
- `image_id`: string - identifier (filename without extension)
- `original_shape`: tuple - original image dimensions

## Step 2: Load Features During Training

Use the `CPSAMFeatureLoader` utility to efficiently load pre-extracted features:

### Basic Usage

```python
from scripts.load_cpsam_features import CPSAMFeatureLoader

# Initialize loader
loader = CPSAMFeatureLoader("data/cpsam_features")

print(f"Loaded {len(loader)} image features")
print(f"Feature dimensionality: {loader.manifest['feature_dim']}")

# Get all available image IDs
image_ids = loader.get_image_ids()

# Load features for a specific image
features = loader.load_features("image_001")  # Returns (256, H, W) array

# Load with metadata
features, metadata = loader.load_features("image_001", return_metadata=True)
print(f"Original shape: {metadata['original_shape']}")
print(f"Feature shape: {features.shape}")
```

### Batch Loading

```python
# Load multiple images at once
batch_ids = ["image_001", "image_002", "image_003"]
features_list = loader.load_batch(batch_ids)

# Or with metadata
features_list, metadata_list = loader.load_batch(batch_ids, return_metadata=True)
```

### Iteration

```python
# Iterate through all features
for image_id, features, metadata in loader:
    print(f"{image_id}: {features.shape}")
    # Use features for training...
```

### Integration with PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader

class DistillationDataset(Dataset):
    def __init__(self, feature_dir, image_dir):
        self.feature_loader = CPSAMFeatureLoader(feature_dir)
        self.image_ids = self.feature_loader.get_image_ids()
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load teacher features
        teacher_features, metadata = self.feature_loader.load_features(
            image_id, return_metadata=True
        )

        # Load original image (for student model input)
        img_path = Path(metadata["image_path"])
        image = io.imread(str(img_path))

        return {
            "image": torch.from_numpy(image).float(),
            "teacher_features": torch.from_numpy(teacher_features).float(),
            "image_id": image_id,
        }

# Create data loader
dataset = DistillationDataset("data/cpsam_features", "data/train_images")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
```

## Step 3: Verify Extracted Features

Verify that features were extracted correctly:

```bash
# Verify all features
python scripts/load_cpsam_features.py data/cpsam_features

# Verify specific number of samples
python scripts/load_cpsam_features.py data/cpsam_features --num-samples 10
```

Expected output:
```
2024-01-15 10:00:00 - INFO - Verifying features in data/cpsam_features
2024-01-15 10:00:00 - INFO -   ✓ Manifest loaded: 1234 images
2024-01-15 10:00:01 - INFO -   ✓ image_001: (256, 64, 64) (original: [512, 512])
2024-01-15 10:00:01 - INFO -   ✓ image_002: (256, 64, 64) (original: [512, 512])
...
2024-01-15 10:00:05 - INFO - Verification passed!
```

## Programmatic Feature Extraction

For custom workflows, you can use the extraction function directly:

```python
from cellpose.models import CellposeModel
from cellpose import io
from scripts.extract_cpsam_features import extract_encoder_features

# Load model
model = CellposeModel(pretrained_model="cpsam", gpu=True)

# Load image
image = io.imread("my_image.png")

# Extract features
features, feature_shape = extract_encoder_features(
    model.net,
    image,
    bsize=256,
    batch_size=8,
    device=model.device,
)

print(f"Extracted features: {features.shape}")  # (256, H, W)
print(f"Feature spatial dims: {feature_shape}")  # (H, W)
```

## Storage Considerations

**Feature file sizes:**
- Each `.npz` file is typically 10-50 MB (depending on image size)
- For 1000 images: ~10-50 GB total storage
- Compression: ~50% smaller than uncompressed `.npy` files

**Recommendations:**
- Extract features on a machine with sufficient disk space
- Use SSD storage for faster I/O during training
- Consider extracting in batches if dataset is very large (10k+ images)

## Troubleshooting

### Out of Memory During Extraction

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `--batch-size` (try 4 or 2)
- Reduce `--bsize` (try 224 or 192)
- Process images on CPU (remove `--gpu` flag, slower but no memory limit)

### Feature Shape Mismatch

If feature dimensions don't match expectations, verify:
- CPSAM model is correctly loaded (check model architecture)
- Image preprocessing is consistent with training
- Encoder neck is producing 256-channel output

### Missing Images

If some images are skipped:
```
WARNING - Skipped invalid items: 5
```

Check:
- Image file format is supported
- Images are not corrupted
- Image files have read permissions

## Best Practices

1. **Extract Once, Train Many**: Feature extraction is slow (forward pass through encoder). Extract once and reuse for multiple training experiments.

2. **Validate After Extraction**: Always run the verification script to ensure features are correctly extracted.

3. **Organize by Dataset**: Keep features organized by dataset:
   ```
   data/
     ├─ cpsam_features_cellpose/
     ├─ cpsam_features_livecell/
     └─ cpsam_features_custom/
   ```

4. **Track Feature Version**: Include model name and date in feature directory names:
   ```
   data/cpsam_v1.0_features_20240115/
   ```

5. **Backup Manifests**: The `manifest.json` is critical for loading features. Keep backups.

## Next Steps

After extracting features, you can:
1. **Train a distilled model** using the extracted features as supervision
2. **Analyze feature distributions** to understand what CPSAM has learned
3. **Compare features** across different datasets or model versions

For student model training with distillation loss, see: *(documentation to be added)*

## References

- **CPSAM Model**: See [FOUNDATION_TRAINING.md](FOUNDATION_TRAINING.md)
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network"
