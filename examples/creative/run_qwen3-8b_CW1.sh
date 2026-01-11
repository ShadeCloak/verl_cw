# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

# hostname -I | awk '{print $1}'
# ray start --head --port=6379 --node-ip-address=100.64.90.242 --num-gpus=8
# ray start --address=100.64.90.242:6379 --num-gpus=8



ray stop --force

wandb login fe3a3f867639b3d57b39d7af7d0527150fc052fe

ray start --head --disable-usage-stats --include-dashboard=False --num-gpus=8


set -x

# NCCL timeout and debug settings
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800  # 30 minutes
# export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ssh node-3 "ray start --address=$(hostname -i):6379 --num-gpus=8 --block" &

# HEAD_IP=100.64.90.242
# NODES=("node-1" "node-2" "node-3")

# # 启动 head 节点
# /home/aiscuser/miniconda3/envs/verl/bin/ray start --head --port=6379 --node-ip-address=$HEAD_IP --num-gpus=8

# # 启动 worker 节点
# for node in "${NODES[@]}"; do
#   ssh -o StrictHostKeyChecking=no $node \
#     "/home/aiscuser/miniconda3/envs/verl/bin/ray start --address=$HEAD_IP:6379 --num-gpus=8 --block" &
# done
# export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# export WANDB_RESUME=never
# export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")

# export WANDB_API_KEY=fe3a3f867639b3d57b39d7af7d0527150fc052fe

# sleep 10
# export PYTHONDONTWRITEBYTECODE=1          # 不再写 .pyc
# export TRANSFORMERS_TRUST_REMOTE_CODE=true
python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +reward_model.style='model' \
    reward_model.enable=True \
    reward_model.model.local_path="/workspace/qingnan/model/Skywork-Reward-V2-Llama-3.1-8B-40M" \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    reward_model.micro_batch_size_per_gpu=64 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_CW_2' \
    trainer.experiment_name="qwen3_8b_function_rm_20251015_070029" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@