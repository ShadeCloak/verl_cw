#!/bin/bash

  echo "=== 开始清理 ==="

  # 1. 停止Ray
  echo "停止Ray集群..."
  /workspace/chrihan/miniconda3/envs/verl_train_sa_1/bin/ray stop --force 2>/dev/null || true

  # 2. 清理进程
  echo "清理Ray进程..."
  pkill -9 -f "ray::" 2>/dev/null || true
  pkill -9 -f "raylet" 2>/dev/null || true
  pkill -9 -f "gcs_server" 2>/dev/null || true
  pkill -9 -f "ray.*dashboard" 2>/dev/null || true
  pkill -9 -f "ray.*monitor" 2>/dev/null || true
  pkill -9 -f "python.*verl" 2>/dev/null || true
  pkill -9 -f "python.*main_ppo" 2>/dev/null || true

  # 等待进程终止
  sleep 3

  # 3. 清理临时文件
  echo "清理临时文件..."
  rm -rf /tmp/ray/*

  # 4. 检查结果
  echo "=== 清理完成 ==="
  echo "剩余Ray进程:"
  ps aux | grep -E "ray|verl" | grep -v grep || echo "无"

  echo "GPU状态:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || echo "无GPU进程"

  echo "僵尸进程:"
  ps aux | grep defunct | grep -v grep || echo "无"