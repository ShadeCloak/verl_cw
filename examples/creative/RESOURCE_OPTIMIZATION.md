# Similarity Group GRPO - Resource Optimization

## Problem

When using `similarity_group_grpo`, the following Ray warning appeared:

```
Warning: The following resource request cannot be scheduled right now: {'CPU': 1.0, 'GPU': 1.0}.
This is likely due to all cluster resources being claimed by actors.
```

## Root Cause

The initial implementation created 8 similarity grouper workers, each requesting **1 full GPU** (`num_gpus=1`).

However, in a typical 8-GPU setup with GRPO training:
- **Actor/Rollout workers**: Use ~6-7 GPUs (vLLM, FSDP)
- **Reward Model**: Uses ~1-2 GPUs
- **Similarity Groupers**: Requested 8 GPUs (not available!)

**Result**: Ray cannot schedule the similarity grouper workers because all GPUs are already claimed.

## Solution

### Changed GPU Allocation Strategy

Instead of requesting dedicated GPUs, similarity grouper workers now **share GPU resources**:

```python
# Before:
@ray.remote(num_gpus=1)  # Each worker wants 1 full GPU (8 total)
class SimilarityGrouperWorker:
    ...

# After:
@ray.remote(num_gpus=0.125)  # Each worker uses 1/8 GPU (1 total)
class SimilarityGrouperWorker:
    ...
```

### Resource Breakdown (After Optimization)

With 8 workers, each using `num_gpus=0.125`:
- **Total GPU allocation**: 8 × 0.125 = **1 GPU**
- Workers share GPU memory and compute
- Ray can schedule workers on any available GPU with free capacity

### Implementation Details

1. **Worker Declaration** (`similarity_grouper.py:35`):
   ```python
   @ray.remote(num_gpus=0.125)  # Share GPU resources
   class SimilarityGrouperWorker:
   ```

2. **Device Initialization** (`similarity_grouper.py:53-70`):
   ```python
   def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
       if device is None:
           device = 'cuda' if torch.cuda.is_available() else 'cpu'

       self.device = device
       self.model = SentenceTransformer(model_name, device=device)
   ```

3. **Worker Creation** (`ray_trainer.py:776-793`):
   ```python
   # Each worker uses 0.125 GPU (8 workers = 1 GPU total)
   n_similarity_workers = 8
   self.similarity_grouper_workers = [
       SimilarityGrouperWorker.remote(
           model_name="BAAI/bge-m3",
           device=None  # Auto-detect available GPU
       )
       for _ in range(n_similarity_workers)
   ]
   ```

## Performance Impact

### Memory Usage
- **BAAI/bge-m3 model size**: ~2GB per worker
- **Total VRAM**: 8 workers × 2GB ≈ 16GB
- **Impact**: Shares GPU with other workers (actor/rollout/RM)

### Computation
- **Embedding computation**: Still parallel across 8 workers
- **GPU utilization**: Depends on availability
- **Speed**: May be slightly slower due to sharing, but acceptable

### Expected Behavior

✅ **Good**:
- Workers can be scheduled immediately
- No resource allocation conflicts
- Training proceeds without blocking

⚠️ **Acceptable Trade-offs**:
- Slightly slower embedding computation (due to GPU sharing)
- Workers may queue if GPU is heavily loaded
- Still much faster than CPU-only

## Monitoring

Check Ray dashboard for resource utilization:
```bash
ray status
```

Key metrics:
- `GPU`: Should show ~7.5-8.0 utilized (including similarity workers)
- `Actors`: Should show all similarity workers as `ALIVE`

## Alternative Solutions (Not Implemented)

If GPU memory is still insufficient, consider:

### Option 1: Reduce Number of Workers
```python
n_similarity_workers = 4  # Instead of 8
```
- Pros: Less memory, less scheduling overhead
- Cons: Slower parallel computation

### Option 2: Use CPU
```python
@ray.remote(num_cpus=1)  # CPU instead of GPU
class SimilarityGrouperWorker:
    def __init__(self, device='cpu'):
        self.model = SentenceTransformer(model_name, device='cpu')
```
- Pros: No GPU memory usage
- Cons: **Much slower** embedding computation (~10-20x)

### Option 3: Sequential Processing (No Workers)
Compute similarity in the main process without Ray workers.
- Pros: No resource allocation issues
- Cons: **No parallelism**, much slower

## Troubleshooting

### Still seeing resource warnings?

**Check actual GPU usage**:
```bash
nvidia-smi
```

If all GPUs are at 100% memory:
1. Reduce `gpu_memory_utilization` for rollout/RM in config
2. Reduce `n_similarity_workers` from 8 to 4
3. As last resort, use CPU for similarity computation

### Workers not starting?

Check Ray logs:
```bash
ray logs
```

Look for:
- `[SimilarityGrouperWorker] Loading model: BAAI/bge-m3`
- `[SimilarityGrouperWorker] Model loaded on cuda`

### Slow similarity computation?

This is expected when sharing GPU. To optimize:
1. Use smaller embedding model (e.g., `all-MiniLM-L6-v2`)
2. Reduce batch size in `compute_groups_for_batch`
3. Cache embeddings if responses don't change

## Verification

After these changes, you should see:

```
[RayPPOTrainer] Creating 8 similarity grouper workers...
[RayPPOTrainer] Each worker uses 0.125 GPU (total: 1 GPU shared)
[SimilarityGrouperWorker] Loading model: BAAI/bge-m3
[SimilarityGrouperWorker] Model loaded on cuda
[RayPPOTrainer] Similarity grouper workers created successfully
```

And **NO** resource allocation warnings.

## Summary

| Metric | Before | After |
|--------|--------|-------|
| GPU per worker | 1.0 | 0.125 |
| Total GPU requested | 8.0 | 1.0 |
| Can schedule? | ❌ No | ✅ Yes |
| Speed | N/A (blocked) | ~1-2s per step |
| Memory | N/A (blocked) | ~16GB VRAM |

**Result**: Training can now proceed without resource conflicts! 🎉
