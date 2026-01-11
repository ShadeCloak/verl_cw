# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams

from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.roles.reward_model_engine.base import BaseRewardModel

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _post_process_outputs(output, is_embedding, return_token_counts=False):
    """Post-process vLLM outputs to extract scores or output_ids.

    Args:
        output: vLLM RequestOutput objects
        is_embedding: Whether this is an embedding model
        return_token_counts: If True, also return token counts for each output

    Returns:
        If return_token_counts=False: List of output texts or token IDs
        If return_token_counts=True: Tuple of (outputs, token_counts)
    """
    if is_embedding:
        # For discriminative models, extract embeddings
        # vLLM doesn't natively support embedding mode like sglang,
        # so this path may need customization based on your model
        raise NotImplementedError("Discriminative reward model not yet supported with vLLM backend")
    else:
        # Extract token counts from vLLM outputs
        token_counts = [len(o.outputs[0].token_ids) for o in output] if output else []

        # For generative models, check if we have text or token_ids
        # If detokenize=True, vLLM returns text directly
        if output and hasattr(output[0].outputs[0], 'text') and output[0].outputs[0].text:
            # detokenize=True: return text strings
            output_text = [o.outputs[0].text for o in output]
            if return_token_counts:
                return output_text, token_counts
            return output_text
        else:
            # detokenize=False: return token IDs
            output_ids = [o.outputs[0].token_ids for o in output]
            if return_token_counts:
                return output_ids, token_counts
            return output_ids


class VLLMRewardModel(BaseRewardModel):
    """vLLM-based reward model for generative reward models.

    This class provides an alternative to SGLangRewardModel for environments
    where sglang is not available. It uses vLLM as the inference backend.
    """

    def __init__(
        self,
        config: RewardModelConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        # model_config might be a DictConfig, not yet initialized as HFModelConfig
        # So we safely access path attribute
        actor_module = model_config.path
        trust_remote_code = model_config.trust_remote_code

        reward_type = self.config.model_type
        if reward_type == "discriminative":
            self.is_embedding = True
            raise NotImplementedError(
                "Discriminative reward model is not yet supported with vLLM backend. "
                "Please use sglang backend or implement embedding support for vLLM."
            )
        elif reward_type == "generative":
            self.is_embedding = False
        else:
            raise ValueError(f"reward type {reward_type} not supported")

        self._init_distributed_env()

        if reward_type == "generative":
            self._init_sampling_params()

        self._init_inference_engine(trust_remote_code, actor_module)

    def _init_distributed_env(self):
        """Initialize distributed environment for vLLM."""
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )

        world_size = dist.get_world_size()
        self._rank = dist.get_rank()

        # For vLLM, we use data parallelism across TP groups
        self.dp_size = world_size // self.tensor_parallel_size
        self.dp_rank = self._rank // self.tensor_parallel_size
        self.tp_rank = self._rank % self.tensor_parallel_size

        if self._rank == 0:
            logger.info(
                f"VLLMRewardModel distributed env: world_size={world_size}, "
                f"tp_size={self.tensor_parallel_size}, dp_size={self.dp_size}"
            )

    def _init_inference_engine(self, trust_remote_code, actor_module):
        """Initialize the vLLM inference engine."""
        # Get engine kwargs from config
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}

        # Determine max model length
        max_model_len = self.config.get("max_model_len", None)
        if max_model_len is None:
            max_model_len = self.config.prompt_length + self.config.response_length

        if self._rank == 0:
            logger.info(f"Initializing vLLM engine with model: {actor_module}")
            logger.info(f"TP size: {self.tensor_parallel_size}, max_model_len: {max_model_len}")

        # With external_launcher, all ranks need to initialize the engine
        # vLLM will handle TP internally
        # CRITICAL FIX: Do NOT skip tokenizer initialization for DeepSeek-GRM
        # The tokenizer is needed to properly handle string inputs
        skip_tokenizer = self.config.get("skip_tokenizer_init", False)
        if "DeepSeek-GRM" in actor_module:
            skip_tokenizer = False  # Force tokenizer initialization for DeepSeek-GRM
            if self._rank == 0:
                logger.info("DeepSeek-GRM detected: forcing tokenizer initialization")

        self._engine = LLM(
            model=actor_module,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.config.dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enforce_eager=self.config.enforce_eager,
            max_model_len=max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            enable_prefix_caching=self.config.enable_prefix_caching,
            disable_log_stats=self.config.disable_log_stats,
            trust_remote_code=trust_remote_code,
            load_format=self.config.load_format,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
            skip_tokenizer_init=skip_tokenizer,
            **engine_kwargs,
        )

        if self._rank == 0:
            logger.info("vLLM engine initialized successfully")

        self.sharding_manager = None
        self.is_sleep = True

    def _init_sampling_params(self):
        """Initialize sampling parameters for generative reward model."""
        kwargs = dict(
            n=1,
            max_tokens=self.config.response_length,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=self.config.get("repetition_penalty", 1.0),
        )

        # Support adding any sampling params from the config file
        vllm_sampling_params = SamplingParams()
        for k in self.config.keys():
            if hasattr(vllm_sampling_params, str(k)) or "stop" in str(k):
                kwargs[k] = self.config.get(k)

        kwargs["n"] = 1  # already repeated in ray_trainer

        # For DeepSeek-GRM with string inputs: use detokenize=True to get text directly
        # This is the recommended approach based on working test.py
        if "DeepSeek-GRM" in self.config.model_config.path:
            kwargs["detokenize"] = True
        else:
            # For other models: use detokenize=False to get token IDs
            kwargs["detokenize"] = False

        self.sampling_params = SamplingParams(**kwargs)

        if self._rank == 0:
            logger.info(f"Sampling params: {self.sampling_params}")

    def compute_reward(self, rm_input_ids):
        """Compute rewards using vLLM inference.

        Args:
            rm_input_ids: List of token ID lists OR strings for reward model inputs

        Returns:
            output_ids: List of generated token ID lists (for generative models)
        """
        # DEBUG: Log input information
        if self._rank == 0 and rm_input_ids:
            import os
            debug_file = "/workspace/qingnan/verl/examples/tmp/vllm_input_debug.log"
            os.makedirs(os.path.dirname(debug_file), exist_ok=True)
            with open(debug_file, 'a') as f:
                f.write(f"\n=== vLLM Input Debug ===\n")
                f.write(f"Number of inputs: {len(rm_input_ids)}\n")
                f.write(f"First input type: {type(rm_input_ids[0])}\n")
                if isinstance(rm_input_ids[0], str):
                    f.write(f"First input length (chars): {len(rm_input_ids[0])}\n")
                    f.write(f"First input (first 12000 chars): {rm_input_ids[0][:12000]}\n")
                else:
                    f.write(f"First input length (tokens): {len(rm_input_ids[0]) if rm_input_ids else 0}\n")
                    f.write(f"First input (first 50 tokens): {rm_input_ids[0][:50] if rm_input_ids else []}\n")
                f.write(f"Sampling params: {self.sampling_params}\n")

        # Prepare inputs for vLLM
        # Check if inputs are strings or token IDs
        if rm_input_ids and isinstance(rm_input_ids[0], str):
            # String inputs - pass directly to vLLM (recommended for DeepSeek-GRM)
            vllm_inputs = rm_input_ids
        else:
            # Token ID inputs - use prompt_token_ids format
            vllm_inputs = [{"prompt_token_ids": input_ids} for input_ids in rm_input_ids]

        # Generate outputs using vLLM
        # With external_launcher, all ranks participate in generation
        if self.is_embedding:
            raise NotImplementedError("Embedding mode not supported for vLLM reward model")
        else:
            vllm_outputs = self._engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

        # Post-process outputs with token counts
        output, token_counts = _post_process_outputs(vllm_outputs, self.is_embedding, return_token_counts=True)

        # Store token counts for later retrieval by reward_model.py
        self._last_token_counts = token_counts

        return output

    async def resume(self, tags: list[str]):
        """Resume reward model weights or kv cache in GPU memory.

        Note: vLLM doesn't have the same memory management as sglang,
        so this is a no-op for now.

        Args:
            tags: weights or kv_cache.
        """
        # vLLM doesn't support the same fine-grained memory management as sglang
        # This is left as a no-op for compatibility
        pass

    async def release(self):
        """Release weights and kv cache in GPU memory.

        Note: vLLM doesn't have the same memory management as sglang,
        so this is a no-op for now.
        """
        # vLLM doesn't support the same fine-grained memory management as sglang
        # This is left as a no-op for compatibility
        pass
