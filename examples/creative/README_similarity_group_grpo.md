# Similarity Group GRPO for Creative Writing

## Overview

This implementation adds similarity-based sub-grouping to GRPO (Group Relative Policy Optimization) for creative writing tasks. The key idea is to preserve diversity in creative writing by only normalizing rewards within groups of similar responses, rather than comparing all responses to the same prompt.

## Motivation

Creative writing differs from math/code tasks in that:
- **Multiple valid approaches**: Different creative angles are all valid
- **Diversity is valuable**: We want to encourage diverse thinking, not penalize it
- **Prompt ambiguity**: Prompts may be intentionally open-ended

Traditional GRPO compares all K responses to the same prompt, which can:
- Suppress creative diversity
- Penalize valid but unique responses
- Lead to mode collapse in creative tasks

## Solution: Two-Level Grouping

### Level 1: Prompt-level grouping
Group responses by prompt (existing behavior)

### Level 2: Similarity-based sub-grouping
Within each prompt's responses, further group by similarity:
1. Extract result text (after `</think>` tag)
2. Compute embeddings using BAAI/bge-m3
3. Use hierarchical clustering + silhouette coefficient to find optimal groups (2-7)
4. If silhouette score < 0.1, treat all as one group (highly similar)

### Reward Normalization
- **Only normalize within similarity sub-groups**
- **Single-member groups get advantage = 0** (no normalization needed)

## Implementation Files

### 1. New Module: `verl/workers/similarity_grouper.py`
- Ray remote worker for parallel similarity computation
- Uses BAAI/bge-m3 on GPU
- Implements silhouette-based optimal clustering
- Functions:
  - `SimilarityGrouperWorker`: Ray worker class
  - `compute_single_member_mask()`: Identify single-member groups
  - `compute_group_distribution_per_prompt()`: Statistics for logging

### 2. Modified: `verl/trainer/ppo/core_algos.py`
- Added `compute_similarity_group_grpo_outcome_advantage()`
- Registered as `"similarity_group_grpo"` advantage estimator
- Uses combined group labels: `"{uid}_{sim_group}"`
- Sets single-member group advantages to 0

### 3. Modified: `verl/trainer/ppo/ray_trainer.py`
- **Initialization** (line 769-785):
  - Creates 8 similarity grouper workers (one per GPU)
  - Only if `algorithm.adv_estimator == "similarity_group_grpo"`

- **Rollout processing** (line 1057-1127):
  - Extracts result text from responses
  - Parallel similarity computation across 8 GPUs
  - Stores `similarity_group_labels` in batch
  - Computes statistics for logging

- **Advantage computation** (line 254-255):
  - Passes `similarity_group_labels` to advantage function
  - Uses new `similarity_group_grpo` estimator

- **Metrics logging** (line 1322-1340):
  - `similarity_groups/n_prompts_split_into_{k}_groups` (k=1..8)
  - `similarity_groups/avg_groups_per_prompt`
  - `similarity_groups/n_single_member_groups`
  - `similarity_groups/avg_silhouette_score`

## Usage

### Configuration

In your training script (e.g., `run_qwen3-8b_CW_deepseek_kl_en_ins_groupRPO.sh`), set:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=similarity_group_grpo \
    # ... other configs ...
```

That's it! The similarity grouping will be automatically applied.

### What Happens During Training

1. **Rollout**: Generate K=8 responses per prompt
2. **Similarity Grouping** (new step):
   - Extract result text from each response
   - Compute embeddings in parallel (8 GPUs)
   - Group by similarity within each prompt
   - Example: 8 responses → 3 groups: [2 responses, 4 responses, 2 responses]
3. **Reward Computation**:
   - Single-member groups can be skipped (advantage will be 0 anyway)
4. **Advantage Calculation**:
   - Normalize only within similarity sub-groups
   - Preserve diversity across different creative approaches
5. **Logging**:
   - Track grouping statistics in wandb

### WandB Metrics

Monitor these metrics to understand grouping behavior:

- `similarity_groups/n_prompts_split_into_1_groups`: How many prompts have all responses similar (potential reward hacking)
- `similarity_groups/n_prompts_split_into_k_groups`: Distribution across k=1..8
- `similarity_groups/avg_groups_per_prompt`: Average diversity per prompt
- `similarity_groups/avg_silhouette_score`: Clustering quality (higher = better separation)

### Expected Behavior

**Good diversity**:
- Most prompts split into 3-5 groups
- Low count of 1-group prompts
- Silhouette score > 0.2

**Potential issues**:
- Many 1-group prompts → responses too similar (check for reward hacking)
- Many 8-group prompts → responses too different (check embedding quality)

## Performance Considerations

### GPU Usage
- Uses 8 parallel workers (one per GPU)
- Each worker loads BAAI/bge-m3 (~2GB VRAM)
- Runs during rollout, adds ~1-2s per step (depending on batch size)

### Optimization Tips
1. **Result extraction**: Already optimized to reuse reward model's logic
2. **Parallel computation**: Work is split evenly across 8 GPUs
3. **Caching**: Embeddings are not cached (computed fresh each step)

## Debugging

### Check if similarity grouping is active:
```bash
# Look for these logs during training:
[RayPPOTrainer] Creating 8 similarity grouper workers...
[SimilarityGrouperWorker] Loading model: BAAI/bge-m3
```

### Verify grouping results:
Check wandb for `similarity_groups/*` metrics

### Common Issues

**Issue**: All prompts grouped as 1
- Check if responses are actually diverse (may indicate reward hacking)
- Lower `MIN_SILHOUETTE` threshold in `similarity_grouper.py:43`

**Issue**: Slow training
- Reduce number of workers from 8 to 4 in `ray_trainer.py:777`
- Use smaller embedding model (modify `model_name` parameter)

**Issue**: Out of memory
- Reduce `n_similarity_workers` in `ray_trainer.py:777`
- Check GPU memory allocation

## Algorithm Details

### Silhouette Coefficient Method

For each prompt with K=8 responses:

1. **Compute similarity matrix**:
   ```python
   embeddings = model.encode(results)
   similarity = cosine_similarity(embeddings)
   ```

2. **Try clustering with 2-7 groups**:
   ```python
   for n_groups in range(2, 8):
       labels = AgglomerativeClustering(n_groups).fit_predict(1 - similarity)
       score = silhouette_score(embeddings, labels)
   ```

3. **Select best grouping**:
   - Choose n_groups with highest silhouette score
   - If best_score < 0.1: collapse to 1 group (all similar)

4. **Normalize advantages**:
   ```python
   # Within each group g:
   advantage[i] = (reward[i] - mean_g) / std_g

   # Single-member groups:
   advantage[i] = 0
   ```

### Why Silhouette Coefficient?

- **Range**: [-1, 1], higher is better
- **Interpretation**: Measures cluster cohesion vs. separation
- **No hyperparameters**: Automatically finds optimal k
- **Robust**: Works well for imbalanced groups

### Alternative Methods (Not Implemented)

Could explore:
- **DBSCAN**: Density-based clustering (may find arbitrary-shaped groups)
- **Threshold-based**: Use similarity > threshold (requires tuning)
- **LLM-based**: Use LLM to judge similarity (too slow)

## Citation

If you use this implementation, please cite:

```bibtex
@misc{similarity_group_grpo,
  title={Similarity Group GRPO for Creative Writing},
  author={Your Name},
  year={2025},
  note={Implementation for preserving diversity in creative writing tasks}
}
```

## Future Work

- [ ] Cache embeddings across steps (if responses don't change much)
- [ ] Experiment with different embedding models
- [ ] Adaptive thresholding based on task type
- [ ] Multi-level grouping (more than 2 levels)
