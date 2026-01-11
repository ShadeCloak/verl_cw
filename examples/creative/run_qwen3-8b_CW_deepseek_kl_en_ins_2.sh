# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

# hostname -I | awk '{print $1}'
# ray start --head --port=6379 --node-ip-address=100.64.90.242 --num-gpus=8
# ray start --address=100.64.90.242:6379 --num-gpus=8


# apt-get update && apt-get install -y git

# # 或者在 CentOS/RHEL 上
# yum install -y git
# 安装 VS Code CLI：
# bash# 下载并安装 VS Code CLI
# curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
# tar -xf vscode_cli.tar.gz
# mv code /usr/local/bin/

# apt install build-essential
# source /workspace/yangwang/.zshrc
# source /workspace/chrihan/miniconda3/etc/profile.d/conda.sh
# conda activate verl_train_sa_1
export RAY_BACKEND_LOG_LEVEL=debug
export PYTHONUNBUFFERED=1

ray stop --force

wandb login fe3a3f867639b3d57b39d7af7d0527150fc052fe

ray start --head --disable-usage-stats --include-dashboard=False --num-gpus=8 --num-cpus=128

# export LD_LIBRARY_PATH=/workspace/yangwang/miniconda3/envs/verl_train_sa/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
# /workspace/chrihan/miniconda3/envs/verl_train_sa/lib/python3.12/site-packages/torch
# source /workspace/yangwang/.zshrc
 
# source /workspace/chrihan/miniconda3/etc/profile.d/conda.sh
export RAY_BACKEND_LOG_LEVEL=debug
export PYTHONUNBUFFERED=1

set -x
# export VLLM_ATTENTION_BACKEND=FLASHINFER
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
export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export WANDB_RESUME=never
export WANDB_RUN_ID=$(/workspace/chrihan/miniconda3/envs/verl_train_sa_1/bin/python3 -c "import wandb; print(wandb.util.generate_id())")

export WANDB_API_KEY=fe3a3f867639b3d57b39d7af7d0527150fc052fe
export RAY_BACKEND_LOG_LEVEL=debug
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=DEBUG
sleep 10
export PYTHONDONTWRITEBYTECODE=1          # 不再写 .pyc
export TRANSFORMERS_TRUST_REMOTE_CODE=true
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
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=300000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model=vllm_reward_model \
    +reward_model.style='model' \
    +reward_model.input_model_config.path="/workspace/qingnan/model/Qwen3-8B" \
    reward_model.model_config.path="/workspace/qingnan/model/DeepSeek-GRM-16B" \
    reward_model.tensor_model_parallel_size=1 \
    reward_model.micro_batch_size_per_gpu=8 \
    reward_model.max_num_seqs=32 \
    reward_model.max_num_batched_tokens=300000 \
    reward_model.gpu_memory_utilization=0.4 \
    reward_model.prompt_length=17000 \
    reward_model.response_length=6000 \
    reward_model.max_model_len=35000 \
    reward_model.enable_chunked_prefill=True \
    reward_model.enable_prefix_caching=True \
    reward_model.enforce_eager=True \
    reward_model.data_processor_config.path=/workspace/qingnan/verl/examples/creative/deepseek_grm_process_fn.py \
    reward_model.data_processor_config.preprocess_fn_name=construct_deepseek_grm_inputs \
    reward_model.data_processor_config.postprocess_fn_name=convert_deepseek_grm_output_to_reward \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_CW_2' \
    trainer.experiment_name="qwen3_8b_function_rm_kl_en_ins_no_think_20251107_032953" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@