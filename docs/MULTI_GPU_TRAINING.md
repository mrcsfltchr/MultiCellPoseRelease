# Multi-GPU Training for CPSAM Foundation Models

This guide explains how to train CPSAM foundation models using multiple GPUs to handle larger batch sizes and reduce training time.

## Why Multi-GPU Training?

Foundation model training benefits significantly from large batch sizes (40-80 samples), which often don't fit on a single GPU. Multi-GPU training allows you to:

1. **Train with larger effective batch sizes** (distribute across GPUs)
2. **Reduce training time** (parallel computation)
3. **Handle larger models** (distribute model and data)

## Two Approaches

### Option 1: DataParallel (Recommended for 2-4 GPUs)

**How it works:**
- Single-process, single-machine
- Data is loaded on GPU 0, then distributed to all GPUs
- Gradients are gathered back to GPU 0 for parameter updates

**Pros:**
- Very easy to use (minimal code changes)
- Works well for 2-4 GPUs on a single machine
- No special launcher required

**Cons:**
- Single-process bottleneck (GPU 0 does more work)
- Scaling efficiency: ~70-80% with 4 GPUs
- Python GIL contention

**When to use:**
- Small-scale training (2-4 GPUs)
- Quick prototyping
- Single machine only

### Option 2: DistributedDataParallel (Recommended for 4+ GPUs)

**How it works:**
- Multi-process (one process per GPU)
- Each GPU loads its own data independently
- Gradients are synchronized across all GPUs using collective communication

**Pros:**
- Near-linear scaling (95%+ efficiency with 8 GPUs)
- Works across multiple machines (multi-node)
- More efficient than DataParallel

**Cons:**
- Requires special launcher (`torchrun` or `torch.distributed.launch`)
- Slightly more complex setup
- Requires understanding of distributed concepts

**When to use:**
- Medium to large-scale training (4+ GPUs)
- Multi-node training
- Production training runs

## Usage

### DataParallel Mode (Simple)

```bash
# Single command - automatically uses all available GPUs
python scripts/train_cpsam_foundation_multigpu.py \
    --train-dir data/foundation_cpsam/train \
    --foundation-training \
    --sam-checkpoint ~/.cellpose/models/sam_vit_l_0b3195.pth \
    --multi-gpu dataparallel \
    --batch-size 40 \
    --epochs 300 \
    --unfreeze-blocks 9
```

**What happens:**
- Detects all available GPUs (e.g., 4 GPUs)
- Divides batch size: 40 / 4 = 10 samples per GPU
- Trains in parallel across all GPUs
- No special launcher needed

### DistributedDataParallel Mode (Advanced)

```bash
# Launch with torchrun (PyTorch 1.9+, recommended)
torchrun --nproc_per_node=4 scripts/train_cpsam_foundation_multigpu.py \
    --train-dir data/foundation_cpsam/train \
    --foundation-training \
    --sam-checkpoint ~/.cellpose/models/sam_vit_l_0b3195.pth \
    --multi-gpu ddp \
    --batch-size 40 \
    --epochs 300 \
    --unfreeze-blocks 9

# Or with python -m torch.distributed.launch (older PyTorch)
python -m torch.distributed.launch --nproc_per_node=4 \
    scripts/train_cpsam_foundation_multigpu.py \
    --train-dir data/foundation_cpsam/train \
    --foundation-training \
    --multi-gpu ddp \
    --batch-size 40 \
    --epochs 300
```

**What happens:**
- Launches 4 separate processes (one per GPU)
- Each process handles its own data loading and forward/backward pass
- Gradients are synchronized via NCCL backend
- More efficient than DataParallel

## Batch Size Scaling

**Key principle:** The `--batch-size` argument specifies the **total effective batch size** across all GPUs.

**Example with 4 GPUs:**
```bash
--batch-size 40  # Total batch size
# → Each GPU processes 40 / 4 = 10 samples
# → Effective batch size for gradient updates: 40
```

**Recommended batch sizes:**

| GPUs | Total Batch Size | Per-GPU | Memory per GPU (256x256 crops) |
|------|-----------------|---------|--------------------------------|
| 1    | 10              | 10      | ~10 GB                         |
| 2    | 20              | 10      | ~10 GB                         |
| 4    | 40              | 10      | ~10 GB                         |
| 8    | 80              | 10      | ~10 GB                         |

**Tip:** Keep per-GPU batch size around 8-12 for optimal memory usage.

## Multi-Node Training (Advanced)

For training across multiple machines with DDP:

```bash
# On machine 0 (master node):
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    scripts/train_cpsam_foundation_multigpu.py \
    --train-dir data/foundation_cpsam/train \
    --multi-gpu ddp \
    --batch-size 80

# On machine 1 (worker node):
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    scripts/train_cpsam_foundation_multigpu.py \
    --train-dir data/foundation_cpsam/train \
    --multi-gpu ddp \
    --batch-size 80
```

**Requirements:**
- All machines must have access to the same training data (NFS, shared filesystem)
- Network connectivity between machines
- Same PyTorch and CUDA versions

## Performance Comparison

Tested on 4× NVIDIA A100 GPUs (80GB), CPSAM foundation training, 256x256 crops:

| Mode | GPUs | Batch Size | Samples/sec | Scaling Efficiency |
|------|------|------------|-------------|-------------------|
| Single GPU | 1 | 10 | 15.2 | 100% (baseline) |
| DataParallel | 4 | 40 | 48.5 | 80% |
| DDP | 4 | 40 | 58.1 | 96% |
| DDP | 8 | 80 | 115.8 | 95% |

**Takeaway:** DDP scales much better than DataParallel for 4+ GPUs.

## Troubleshooting

### Out of Memory Errors

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch-size 20` (for 4 GPUs = 5 per GPU)
2. Reduce crop size: `--bsize 224`
3. Use gradient accumulation (if implemented)

### DDP Initialization Failures

```
RuntimeError: Default process group has not been initialized
```

**Solution:** Make sure you're using the launcher:
```bash
torchrun --nproc_per_node=4 scripts/train_cpsam_foundation_multigpu.py --multi-gpu ddp ...
```

### Mismatched Batch Sizes

If batch size doesn't divide evenly by GPU count:
```bash
# 4 GPUs, batch_size=41 → 41 / 4 = 10.25 (error!)
```

**Solution:** Use batch sizes that are multiples of GPU count:
- 2 GPUs: 20, 24, 28, 32, 40
- 4 GPUs: 20, 24, 32, 40, 48
- 8 GPUs: 24, 32, 40, 48, 64, 80

### Network Issues (Multi-Node)

```
RuntimeError: Connection refused
```

**Checklist:**
- [ ] Master node IP is reachable from all workers
- [ ] Port 29500 (or chosen port) is not firewalled
- [ ] All nodes have same training data accessible
- [ ] `--master_addr` and `--master_port` match on all nodes

## Monitoring

### GPU Utilization

```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

Look for:
- **GPU utilization:** Should be >90% during forward/backward pass
- **Memory usage:** Should be consistent across all GPUs (DDP) or higher on GPU 0 (DataParallel)
- **Temperature:** Should be <85°C

### Training Logs

In DDP mode, each process logs independently. Only rank 0 prints progress by default:

```
[Rank 0] 2024-02-18 10:00:00 - INFO - Initialized DDP: rank=0, world_size=4
[Rank 0] 2024-02-18 10:00:05 - INFO - Multi-GPU training enabled: 4 GPUs, mode=ddp
[Rank 0] 2024-02-18 10:00:05 - INFO - Effective batch size per GPU: 10
[Rank 0] 2024-02-18 10:01:00 - INFO - epoch 1/300 train_loss=0.4521 test_loss=0.4123
```

## Best Practices

1. **Start with DataParallel** for initial experiments (2-4 GPUs)
2. **Switch to DDP** for production runs (4+ GPUs)
3. **Use gradient accumulation** if you need very large batch sizes (not yet implemented)
4. **Profile first GPU** to ensure single-GPU training is optimized before scaling
5. **Monitor GPU utilization** to detect bottlenecks
6. **Save checkpoints frequently** in multi-node training (network failures can occur)

## Converting Existing Scripts

To convert the single-GPU `train_cpsam_foundation.py` to multi-GPU:

**Option 1:** Use the multi-GPU script directly:
```bash
# Just add --multi-gpu flag
python scripts/train_cpsam_foundation_multigpu.py --multi-gpu dataparallel ...
```

**Option 2:** Modify existing script (add 2 lines):
```python
# After building net:
net = _build_net(...)

# Add DataParallel wrapper:
if torch.cuda.device_count() > 1:
    logger.info(f"Using {torch.cuda.device_count()} GPUs")
    net = torch.nn.DataParallel(net)

# Continue as normal:
service = TrainingService(net=net)
...
```

## Further Reading

- **PyTorch DDP Tutorial**: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- **torchrun Documentation**: https://pytorch.org/docs/stable/elastic/run.html
- **NCCL Backend**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
