# DeepSeek-GRM Reward Model 配置指南

## 配置参数位置

### 1. Shell 脚本配置（优先级最高）
文件：`/workspace/qingnan/verl/examples/creative/run_qwen3-8b_CW_deepseek.sh`

### 2. YAML 默认配置
文件：`/workspace/qingnan/verl/verl/trainer/config/reward_model/vllm_reward_model.yaml`

### 3. Python 配置类
文件：`/workspace/qingnan/verl/verl/workers/config/reward_model.py` (RewardModelConfig)

---

## 关键配置参数详解

### 并行配置

#### `tensor_model_parallel_size` (TP)
- **当前值**: 未设置（默认 1）
- **作用**: 张量并行，将模型分片到多个 GPU
- **推荐值**:
  - DeepSeek-GRM-16B: **1** 或 **2**（单卡装不下用 2）
  - 如果你有 8 个 GPU，RM 专用 2 个，可以设置 TP=1 或 2
- **计算公式**: `实际使用的 GPU 数 = tensor_model_parallel_size`
- **示例**:
  ```bash
  reward_model.tensor_model_parallel_size=1    # 单卡推理
  reward_model.tensor_model_parallel_size=2    # 两卡并行
  ```

#### `data_parallel_size`
- **当前值**: 未使用（自动计算）
- **计算**: `dp_size = world_size // tensor_model_parallel_size`
- **说明**: DP 是自动推导的，不需要手动设置

---

### Batch Size 配置

#### `micro_batch_size_per_gpu`
- **当前值**: 未设置（默认 32）
- **作用**: 每个 GPU 处理的样本数
- **推荐值**: **16-64**（根据显存调整）
- **影响**: 越大吞吐量越高，但显存占用越多
- **示例**:
  ```bash
  reward_model.micro_batch_size_per_gpu=32     # 中等
  reward_model.micro_batch_size_per_gpu=64     # 大 batch（需要更多显存）
  ```

#### `max_num_seqs`
- **当前值**: 未设置（默认 64）
- **作用**: vLLM 引擎同时处理的最大序列数
- **推荐值**: **64-256**
- **说明**: 控制 vLLM 的并发能力
- **示例**:
  ```bash
  reward_model.max_num_seqs=128                # 提高并发
  ```

#### `max_num_batched_tokens`
- **当前值**: 未设置（默认 8192）
- **作用**: 单次推理批处理的最大 token 数
- **推荐值**: **8192-16384**（取决于序列长度）
- **计算参考**:
  - 如果 `prompt_length=17000, response_length=4096`
  - 总长度约 21000 tokens/样本
  - `max_num_batched_tokens=21000` 意味着每次只能处理 1 个样本
  - 建议设置为 **40000-80000** 以提高吞吐量
- **示例**:
  ```bash
  reward_model.max_num_batched_tokens=50000    # 允许 2-3 个长序列同时处理
  ```

---

### 显存配置

#### `gpu_memory_utilization`
- **当前值**: 0.3（30%）
- **作用**: vLLM 使用的 GPU 显存比例
- **推荐值**:
  - **单卡部署**: 0.5-0.7
  - **多模型共存**: 0.3-0.4（当前设置合理）
- **说明**:
  - Actor 用 0.4
  - RM 用 0.3
  - 总共 0.7，留 0.3 给系统和其他开销
- **示例**:
  ```bash
  reward_model.gpu_memory_utilization=0.3      # 保持当前值
  ```

---

### 序列长度配置

#### `prompt_length`
- **当前值**: 17000
- **作用**: 输入给 RM 的最大长度（question + response + prompt template）
- **组成**:
  - 原始 question: ~1024 tokens
  - 原始 response: ~15000 tokens
  - DeepSeek-GRM prompt template: ~1500 tokens
  - 总计: ~17500 tokens
- **推荐**: **17000-20000** ✓ 当前设置合理

#### `response_length`
- **当前值**: 4096 ✓（已调整）
- **作用**: RM 生成的评价文本最大长度
- **推荐**: **2048-4096**
- **说明**: DeepSeek-GRM 输出格式：
  ```
  Specific Criteria: ... (200-500 tokens)
  Analysis: ... (500-2000 tokens)
  Score: \boxed{8} (10 tokens)
  ```
  总计约 1000-3000 tokens
- **推荐**: **4096** ✓ 当前已经合理

#### `max_model_len`
- **当前值**: 未设置（默认 32768）
- **作用**: vLLM 引擎的最大上下文长度
- **推荐**: `prompt_length + response_length = 21000-24000`
- **示例**:
  ```bash
  reward_model.max_model_len=24576             # 17000 + 4096 + buffer
  ```

---

### 性能优化配置

#### `enable_chunked_prefill`
- **当前值**: True（默认）
- **作用**: 启用分块预填充，减少延迟
- **推荐**: **True** ✓

#### `enable_prefix_caching`
- **当前值**: True（默认）
- **作用**: 缓存相同的 prompt 前缀
- **推荐**: **True** ✓（提示词模板固定，很有用）

#### `enforce_eager`
- **当前值**: True（默认）
- **作用**: 禁用 CUDA graph（调试模式）
- **推荐**:
  - **调试阶段**: True
  - **生产环境**: False（更快）

---

### 采样配置（generative RM 专用）

#### `sampling_config.temperature`
- **当前值**: 未设置（默认 0.0）
- **推荐**: **0.0**（确定性评分）或 **1.0**（多样性评分）
- **说明**:
  - 0.0 = greedy decoding（每次评分一致）
  - 1.0 = 随机采样（可用于 inference-time scaling）

#### `sampling_config.max_tokens`
- **自动设置**: = `response_length`
- **当前**: 4096 ✓

---

## 推荐配置（针对你的场景）

### 场景：8 个 GPU，Actor + Rollout + Ref + RM 共存

```bash
# 在 run_qwen3-8b_CW_deepseek.sh 中添加/修改：

reward_model=vllm_reward_model \
reward_model.model_config.path="/workspace/qingnan/model/DeepSeek-GRM-16B" \

# ========== 并行配置 ==========
reward_model.tensor_model_parallel_size=1 \        # 单卡 RM（如果显存不够改为 2）

# ========== Batch Size ==========
reward_model.micro_batch_size_per_gpu=32 \         # 根据显存调整（16/32/64）
reward_model.max_num_seqs=128 \                    # vLLM 并发能力
reward_model.max_num_batched_tokens=50000 \        # 允许 2-3 个长序列并行

# ========== 序列长度 ==========
reward_model.prompt_length=17000 \                 # 输入长度（保持不变）
reward_model.response_length=4096 \                # RM 生成长度（已调整）
reward_model.max_model_len=24576 \                 # 总上下文长度

# ========== 显存管理 ==========
reward_model.gpu_memory_utilization=0.3 \          # 30% 显存（与 actor 0.4 共存）

# ========== 性能优化 ==========
reward_model.enable_chunked_prefill=True \
reward_model.enable_prefix_caching=True \
reward_model.enforce_eager=True \                  # 调试用；生产改 False
reward_model.disable_log_stats=True \
reward_model.skip_tokenizer_init=True \

# ========== 数据处理 ==========
reward_model.data_processor_config.path=/workspace/qingnan/verl/examples/creative/deepseek_grm_process_fn.py \
reward_model.data_processor_config.preprocess_fn_name=construct_deepseek_grm_inputs \
reward_model.data_processor_config.postprocess_fn_name=convert_deepseek_grm_output_to_reward \

# ========== 采样配置（可选）==========
reward_model.sampling_config.temperature=0.0 \     # 确定性评分
reward_model.sampling_config.top_p=1.0 \
reward_model.sampling_config.top_k=-1 \
```

---

## 显存分配估算

### DeepSeek-GRM-16B 单卡部署（TP=1）
- **模型权重**: ~32 GB (BF16)
- **KV Cache**: ~5-10 GB（取决于 batch size 和序列长度）
- **总需求**: ~40-45 GB
- **结论**: **需要 A100 80GB 或 H100**

### DeepSeek-GRM-16B 双卡部署（TP=2）
- **每卡权重**: ~16 GB
- **每卡 KV Cache**: ~5-10 GB
- **总需求**: ~25-30 GB/卡
- **结论**: **A100 40GB 可用**

---

## 性能调优建议

### 1. 提高吞吐量
```bash
reward_model.micro_batch_size_per_gpu=64 \         # 增大 batch size
reward_model.max_num_seqs=256 \                    # 提高并发
reward_model.max_num_batched_tokens=100000 \       # 允许更多 token
reward_model.gpu_memory_utilization=0.5 \          # 增加显存使用
```

### 2. 降低显存占用
```bash
reward_model.tensor_model_parallel_size=2 \        # 启用 TP
reward_model.micro_batch_size_per_gpu=16 \         # 减小 batch size
reward_model.max_num_seqs=32 \                     # 降低并发
reward_model.gpu_memory_utilization=0.25 \         # 减少显存
```

### 3. 降低延迟
```bash
reward_model.enforce_eager=False \                 # 启用 CUDA graph
reward_model.enable_chunked_prefill=True \
reward_model.enable_prefix_caching=True \
```

---

## 常见问题

### Q1: 如何确定需要多少个 GPU 给 RM？
**A**: 公式：`RM GPU 数 = tensor_model_parallel_size * data_parallel_size`
- TP=1, DP=1 → 1 个 GPU
- TP=2, DP=1 → 2 个 GPU
- TP=1, DP=2 → 2 个 GPU（数据并行，提高吞吐）

### Q2: OOM 怎么办？
**A**: 调整以下参数（优先级从高到低）：
1. 增加 `tensor_model_parallel_size`（2 → 4）
2. 减小 `micro_batch_size_per_gpu`（32 → 16）
3. 减小 `max_num_batched_tokens`（50000 → 30000）
4. 减小 `gpu_memory_utilization`（0.3 → 0.25）

### Q3: 吞吐量太低怎么办？
**A**:
1. 增加 `micro_batch_size_per_gpu`
2. 增加 `max_num_batched_tokens`
3. 设置 `enforce_eager=False`（启用 CUDA graph）
4. 检查 `prompt_length` 和 `response_length` 是否设置过大

---

## 监控指标

训练时关注以下日志：
```
# RM 吞吐量
[RewardModel] Throughput: XX samples/sec

# 显存使用
[vLLM] GPU memory usage: XX.X GB / XX.X GB

# RM 推理时间
[RewardModel] Inference time: XX.XX sec/batch
```
