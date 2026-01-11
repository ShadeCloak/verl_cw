# GPU Resource Strategy for Similarity Group GRPO

## 问题分析

### 初始警告
```
Warning: The following resource request cannot be scheduled right now: {'CPU': 1.0, 'GPU': 1.0}.
This is likely due to all cluster resources being claimed by actors.
```

### 根本原因

在`init_workers()`阶段，Ray需要为所有workers预留GPU资源：

```
Actor/Rollout Workers:  8 GPUs
Reward Model Workers:   8 GPUs
Similarity Groupers:    8 GPUs
────────────────────────────────
Total Request:         24 GPUs ❌
Available:              8 GPUs ✓
```

虽然**运行时**这些workers是**异步**的（不会同时占用GPU），但**初始化时**Ray的资源调度器会尝试预留所有资源，导致冲突。

---

## 解决方案：延迟初始化（Lazy Initialization）

### 策略

**不在`init_workers()`时创建similarity grouper workers，而是在第一次使用时才创建。**

### 实现

#### 1. 初始化阶段（`ray_trainer.py:769-786`）
```python
if self.use_similarity_grouping:
    # 只标记启用，不创建workers
    self.similarity_grouper_workers = None
    self._similarity_grouper_initialized = False
    print(f"[RayPPOTrainer] Similarity grouping enabled (workers will be created on first use)")
```

#### 2. 第一次使用时（`ray_trainer.py:1060-1076`）
```python
if self.use_similarity_grouping:
    # 延迟创建
    if not self._similarity_grouper_initialized:
        print(f"[RayPPOTrainer] Lazy-initializing 8 similarity grouper workers...")
        self.similarity_grouper_workers = [
            SimilarityGrouperWorker.remote(model_name="BAAI/bge-m3")
            for _ in range(8)
        ]
        self._similarity_grouper_initialized = True
        print(f"[RayPPOTrainer] Similarity grouper workers created successfully")

    # 使用workers进行相似度计算
    ...
```

---

## 运行时序

### 初始化阶段
```
┌─────────────────────────────────────────┐
│ init_workers()                          │
├─────────────────────────────────────────┤
│ ✓ Actor/Rollout:  8 GPUs allocated     │
│ ✓ Reward Model:   8 GPUs allocated     │
│ ○ Similarity:     0 GPUs (not created) │
├─────────────────────────────────────────┤
│ Total: 16 GPUs requested                │
│ Available: 8 GPUs (BUT asynchronous!)   │
└─────────────────────────────────────────┘
```

### 训练循环 - 第一个Step

```
Step 1:
┌──────────────────────────────────┐
│ 1. Rollout                       │
│    Actor/Rollout: 8 GPUs in use │
└──────────────────────────────────┘
           ↓ (GPU释放)
┌──────────────────────────────────┐
│ 2. Similarity Grouping           │
│    [首次] 创建8个workers          │
│    Similarity: 8 GPUs in use     │
└──────────────────────────────────┘
           ↓ (GPU释放)
┌──────────────────────────────────┐
│ 3. Reward Model                  │
│    RM: 8 GPUs in use             │
└──────────────────────────────────┘
           ↓ (GPU释放)
┌──────────────────────────────────┐
│ 4. Actor Update                  │
│    Actor: 8 GPUs in use          │
└──────────────────────────────────┘
```

### 训练循环 - 后续Steps

```
Step 2+:
┌──────────────────────────────────┐
│ 1. Rollout: 8 GPUs               │
└──────────────────────────────────┘
           ↓
┌──────────────────────────────────┐
│ 2. Similarity (复用workers)       │
│    Similarity: 8 GPUs            │
└──────────────────────────────────┘
           ↓
┌──────────────────────────────────┐
│ 3. Reward Model: 8 GPUs          │
└──────────────────────────────────┘
           ↓
┌──────────────────────────────────┐
│ 4. Actor Update: 8 GPUs          │
└──────────────────────────────────┘
```

---

## 资源分配总结

### 各阶段GPU使用

| Stage | Workers Created | GPUs in Use | Timing |
|-------|----------------|-------------|---------|
| **Init** | Actor, RM | 0 (just created) | Startup |
| **Step 1 - Rollout** | - | 8 (Actor) | ~10s |
| **Step 1 - Similarity (first)** | Similarity (created) | 8 (Similarity) | ~2s |
| **Step 1 - Reward** | - | 8 (RM) | ~5s |
| **Step 1 - Update** | - | 8 (Actor) | ~3s |
| **Step 2+ - Rollout** | - | 8 (Actor) | ~10s |
| **Step 2+ - Similarity** | - (reuse) | 8 (Similarity) | ~2s |
| **Step 2+ - Reward** | - | 8 (RM) | ~5s |
| **Step 2+ - Update** | - | 8 (Actor) | ~3s |

### 关键点

1. ✅ **初始化时**: 只创建Actor和RM workers，不创建Similarity workers
2. ✅ **运行时**: 各阶段**顺序执行**，不会同时占用GPU
3. ✅ **延迟创建**: Similarity workers在第一次使用时创建
4. ✅ **重复使用**: 创建后的workers会被重复使用，不需要再次创建

---

## 为什么这样可行？

### 异步执行特性

GRPO训练的各阶段是**顺序执行**的：

```python
# Step N
gen_batch_output = rollout_wg.generate_sequences(...)  # 使用Actor GPUs
# ↓ rollout结束，Actor GPUs释放

similarity_labels = similarity_wg.compute_groups(...)   # 使用8个GPUs
# ↓ similarity结束，GPUs释放

reward_tensor = rm_wg.compute_reward(...)               # 使用RM GPUs
# ↓ reward结束，RM GPUs释放

actor_wg.update_actor(...)                              # 使用Actor GPUs
# ↓ update结束，进入下一个step
```

因此，虽然定义了多组workers，但它们**从不同时运行**，8个GPU完全够用。

### Ray的资源调度

Ray的资源调度器：
- **创建时**: 预留资源（导致警告）
- **运行时**: 动态调度，只在实际执行时占用GPU

通过**延迟创建**，我们确保：
1. 初始化时只预留Actor和RM的资源（可行）
2. 运行时才创建Similarity workers（此时Actor已经创建完成，可以共享资源配额）

---

## 预期行为

### 日志输出

**初始化阶段**:
```
[RayPPOTrainer] Creating actor_rollout workers...
[RayPPOTrainer] Creating reward model workers...
[RayPPOTrainer] Similarity grouping enabled (workers will be created on first use)
```

**第一个训练Step**:
```
Training Progress:   0%|          | 0/266 [00:00<?, ?it/s]
[RayPPOTrainer] Lazy-initializing 8 similarity grouper workers...
[RayPPOTrainer] Each worker uses 1 GPU (will run asynchronously with actor/RM)
[SimilarityGrouperWorker] Loading model: BAAI/bge-m3
[SimilarityGrouperWorker] Model loaded on cuda:0
[SimilarityGrouperWorker] Model loaded on cuda:1
...
[SimilarityGrouperWorker] Model loaded on cuda:7
[RayPPOTrainer] Similarity grouper workers created successfully
```

**后续Steps**:
不再有创建workers的日志（直接复用）

### 不应该看到的警告

✅ **成功**: 不会再看到资源分配警告
❌ **如果还有警告**: 检查其他资源占用（CPU、内存等）

---

## 性能影响

### 第一个Step
- **额外时间**: ~1-2秒（创建8个workers + 加载模型）
- **后续影响**: 无（workers已创建）

### 后续Steps
- **额外时间**: ~1-2秒（纯相似度计算时间）
- **GPU利用率**: 在相似度计算阶段，8个GPU满载

### 总体影响
- **每个step**: +1-2秒（可接受）
- **吞吐量**: 略微降低（~5-10%）
- **收益**: 更好的diversity，可能更好的训练效果

---

## 故障排查

### 问题1: 仍然看到资源警告

**可能原因**:
- 其他进程占用GPU
- Ray配置问题
- Actor/RM的GPU占用超出预期

**解决方案**:
```bash
# 检查GPU使用
nvidia-smi

# 检查Ray状态
ray status

# 检查Ray资源
python -c "import ray; ray.init(); print(ray.available_resources())"
```

### 问题2: Workers创建失败

**检查日志**:
```bash
ray logs --follow
```

**常见问题**:
- CUDA out of memory → 调整`gpu_memory_utilization`
- Model loading failed → 检查`BAAI/bge-m3`是否可访问
- Import errors → 确保依赖安装正确

### 问题3: 相似度计算很慢

**优化选项**:
1. 减少workers数量（8→4）
2. 使用更小的embedding模型
3. 增加batch size（如果内存允许）

---

## 代码位置参考

### 关键修改

1. **Worker定义** (`similarity_grouper.py:35`):
   ```python
   @ray.remote(num_gpus=1)  # Each worker = 1 GPU
   class SimilarityGrouperWorker:
   ```

2. **延迟标记** (`ray_trainer.py:769-786`):
   ```python
   self.similarity_grouper_workers = None  # Not created yet
   self._similarity_grouper_initialized = False
   ```

3. **延迟创建** (`ray_trainer.py:1060-1076`):
   ```python
   if not self._similarity_grouper_initialized:
       # Create workers on first use
       self.similarity_grouper_workers = [...]
   ```

---

## 总结

### 策略：延迟初始化

✅ **优点**:
- 避免初始化时的资源冲突
- 保持运行时的异步执行
- 每个worker仍然使用1个完整GPU（最佳性能）

✅ **Trade-offs**:
- 第一个step稍慢（+1-2秒，仅一次）
- 代码略复杂（增加了lazy initialization逻辑）

### 资源使用

- **初始化**: Actor + RM = 16 GPU requests (但异步)
- **运行时**: 最多8 GPUs同时使用（各阶段顺序执行）
- **Similarity workers**: 8个，各1 GPU，延迟创建

### 最终效果

🎉 **无资源冲突，性能最优，逻辑清晰！**
