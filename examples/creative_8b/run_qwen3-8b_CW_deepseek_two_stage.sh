#!/bin/bash
# Example training script using Two-Stage GRM for creative writing
# Two-Stage GRM: P(principle|question) * P(judge|question, principle, prediction)
# - Stage 1: Generate evaluation principles based on question only (shared across responses)
# - Stage 2: Evaluate each response using the shared principles as generation prefix
#
# Key benefit: Responses to the same question share the same evaluation principles,
# ensuring consistent and fair comparison within the same prompt group.

# Stop on error
set -e

# Ray and environment setup
export RAY_BACKEND_LOG_LEVEL=debug
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=1800

# Clean up any existing Ray instances
ray stop --force

# Login to wandb
wandb login fe3a3f867639b3d57b39d7af7d0527150fc052fe

# Start Ray cluster
ray start --head --disable-usage-stats --include-dashboard=False --num-gpus=8 --num-cpus=64

# Generate unique run ID for this experiment
export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Resume from existing checkpoint - set WANDB_RESUME to allow
export WANDB_RESUME=allow
export WANDB_RUN_ID=$(/root/miniconda3/envs/verl/bin/python3 -c "import wandb; print(wandb.util.generate_id())")
export WANDB_API_KEY=fe3a3f867639b3d57b39d7af7d0527150fc052fe

echo "Starting Two-Stage GRM GRPO training..."
echo "Run ID: $WANDB_RUN_ID"
echo "Timestamp: $TIMESTAMP"

# Run training with Two-Stage GRM
/root/miniconda3/envs/verl/bin/python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/qingnan/verl_1214/data/train.parquet \
    data.val_files=/data/qingnan/verl_1214/data/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=15000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/data/qingnan/model/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.27 \
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
    +reward_model.input_model_config.path="/data/qingnan/model/Qwen3-8B" \
    +reward_model.two_stage_grm=True \
    +reward_model.data_processor_config.template_version="v6" \
    +reward_model.data_processor_config.strip_think_tag=True \
    reward_model.model_config.path="/data/qingnan/DeepSeek-GRM-16B_v1_sft_template2/global_step_68/huggingface" \
    reward_model.tensor_model_parallel_size=1 \
    reward_model.micro_batch_size_per_gpu=4 \
    reward_model.max_num_seqs=32 \
    reward_model.max_num_batched_tokens=800000 \
    reward_model.gpu_memory_utilization=0.55 \
    reward_model.prompt_length=17000 \
    reward_model.response_length=13000 \
    reward_model.max_model_len=40000 \
    reward_model.enable_chunked_prefill=True \
    reward_model.enable_prefix_caching=True \
    reward_model.enforce_eager=True \
    reward_model.data_processor_config.path=/data/qingnan/verl_1214/examples/creative/deepseek_grm_process_fn.py \
    reward_model.data_processor_config.preprocess_fn_name=construct_deepseek_grm_inputs \
    reward_model.data_processor_config.postprocess_fn_name=convert_deepseek_grm_output_to_reward \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_CW_2' \
    trainer.experiment_name="qwen3_8b_two_stage_grm_${TIMESTAMP}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.default_local_dir=/data/qingnan/temp/verl_grpo_example_CW_2/qwen3_8b_two_stage_grm_${TIMESTAMP} \
    trainer.resume_mode=auto $@

echo "Training completed!"

# Cleanup
ray stop --force

# Two-Stage GRM Mode Explanation:
# ================================
# Traditional: For each (question, response) pair, call GRM once to generate:
#   P(principle * judge | question * prediction)
#
# Two-Stage: Split into two calls:
#   Stage 1: P(principle | question) - Generate evaluation principles for the question
#   Stage 2: P(judge | question * principle * prediction) - Use principles as prefix
#
# Benefits:
# 1. Responses to the same question share the same evaluation principles
# 2. More consistent scoring within the same prompt group
# 3. Reduces variance in GRPO's group-wise normalization
# 4. Better aligned with how human evaluators would approach scoring
