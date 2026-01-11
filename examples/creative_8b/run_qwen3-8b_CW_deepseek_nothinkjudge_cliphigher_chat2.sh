#!/bin/bash
# Example training script using Similarity Group GRPO for creative writing
# This script demonstrates how to use the similarity-based sub-grouping feature

# Stop on error
set -e

# Ray and environment setup
export RAY_BACKEND_LOG_LEVEL=debug
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=1800
# export TRANSFORMERS_TRUST_REMOTE_CODE=true
# export PYTHONDONTWRITEBYTECODE=1

# Clean up any existing Ray instances
ray stop --force

# Login to wandb
wandb login fe3a3f867639b3d57b39d7af7d0527150fc052fe

# Start Ray cluster
ray start --head --disable-usage-stats --include-dashboard=False --num-gpus=8 --num-cpus=64

# Wait for Ray to initialize
# sleep 10

# Generate unique run ID for this experiment
export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export WANDB_RESUME=never
export WANDB_RUN_ID=$(/workspace/chrihan/miniconda3/envs/verl_train_sa_1/bin/python3 -c "import wandb; print(wandb.util.generate_id())")
export WANDB_API_KEY=fe3a3f867639b3d57b39d7af7d0527150fc052fe

echo "Starting Similarity Group GRPO training..."
echo "Run ID: $WANDB_RUN_ID"
echo "Timestamp: $TIMESTAMP"

# Run training with Similarity Group GRPO
/workspace/chrihan/miniconda3/envs/verl_train_sa_1/bin/python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/qingnan/verl/data/train.parquet \
    data.val_files=/workspace/qingnan/verl/data/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=15000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/workspace/qingnan/model/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=300000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model=vllm_reward_model \
    +reward_model.style='model' \
    +reward_model.input_model_config.path="/workspace/qingnan/model/Qwen3-8B" \
    +reward_model.data_processor_config.template_version="v0" \
    +reward_model.data_processor_config.strip_think_tag=True \
    reward_model.model_config.path="/workspace/qingnan/model/DeepSeek-GRM-16B" \
    reward_model.tensor_model_parallel_size=1 \
    reward_model.micro_batch_size_per_gpu=4 \
    reward_model.max_num_seqs=16 \
    reward_model.max_num_batched_tokens=300000 \
    reward_model.gpu_memory_utilization=0.4 \
    reward_model.prompt_length=17000 \
    reward_model.response_length=6000 \
    reward_model.max_model_len=30000 \
    reward_model.enable_chunked_prefill=True \
    reward_model.enable_prefix_caching=True \
    reward_model.enforce_eager=True \
    reward_model.data_processor_config.path=/workspace/qingnan/verl/examples/creative/deepseek_grm_process_fn.py \
    reward_model.data_processor_config.preprocess_fn_name=construct_deepseek_grm_inputs \
    reward_model.data_processor_config.postprocess_fn_name=convert_deepseek_grm_output_to_reward \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_CW_2' \
    trainer.experiment_name="qwen3_8b_deepseek_nothinkjudge_cliphigher_chat2_${TIMESTAMP}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@

echo "Training completed!"

# Cleanup
ray stop --force

# Monitor key metrics in wandb:
# - similarity_groups/n_prompts_split_into_1_groups (low is better - indicates diversity)
# - similarity_groups/n_prompts_split_into_k_groups (k=2-7)
# - similarity_groups/avg_groups_per_prompt (should be 3-5 for good diversity)
# - similarity_groups/avg_silhouette_score (>0.2 is good quality clustering)
# - similarity_groups/n_single_member_groups (unique responses, don't participate in normalization)
