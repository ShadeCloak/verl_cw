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
"""
The main entry point to run the PPO algorithm
"""

import logging
import os

import torch

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import (
    get_device_name,
    get_torch_device,
)
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import DistProfiler, DistProfilerExtension, log_gpu_memory_usage
from verl.workers.config import HFModelConfig, RewardModelConfig, RewardModelDataProcessorConfig
from verl.workers.roles.reward_model_engine import get_reward_model_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class RewardModelWorker(Worker, DistProfilerExtension):
    def __init__(self, config: RewardModelConfig) -> None:
        self.config = config
        self.model_config = config.model_config
        self.input_model_config = config.input_model_config
        self.model_type = config.model_type
        assert self.model_type in ["discriminative", "generative"], f"model_type: {self.model_type} is not supported"

        # Check if pairwise_v1 mode is enabled
        self.pairwise_v1 = config.get("pairwise_v1", False)
        if self.pairwise_v1:
            logger.info("Pairwise V1 mode enabled for reward model")

        Worker.__init__(self)
        self.profiler_config = self.config.profiler
        tool_config = self.profiler_config.tool_config
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=tool_config)
        )

        initialize_global_process_group_ray(timeout_second=None)

    def _build_reward_model(self):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse reward model and huggingface model config
        reward_model_config: RewardModelConfig = self.config
        model_config: HFModelConfig = self.config.model_config
        data_processor_config: RewardModelDataProcessorConfig = self.config.data_processor_config
        self.data_processor_config = self.config.data_processor_config
        # self.tokenizer = self.model_config.get_processor()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.path,
            trust_remote_code=True
        )

        # Check if input_model_config has a valid path (handles both None and partial configs)
        input_model_path = None
        if self.input_model_config is not None:
            input_model_path = self.input_model_config.get("path", None)

        if input_model_path is None or input_model_path == "":
            self._do_switch_chat_template = False
            self.src_tokenizer = self.tokenizer
        else:
            self._do_switch_chat_template = True
            # Load tokenizer from input model path
            # Handle both HFModelConfig instances and DictConfig from command line
            if hasattr(self.input_model_config, 'get_processor'):
                self.src_tokenizer = self.input_model_config.get_processor()
            else:
                # For command-line configs, manually load the tokenizer
                self.src_tokenizer = AutoTokenizer.from_pretrained(
                    input_model_path,
                    trust_remote_code=True
                )
        # self.preprocess_fn, self.postprocess_fn = data_processor_config.get_process_fn()

        import importlib.util
        import os

        def load_fn(py_file, fn_name):
            # 动态加载 py_file 模块
            spec = importlib.util.spec_from_file_location("data_proc", py_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, fn_name)

        dp_cfg = self.data_processor_config  # 这是 DictConfig
        self.preprocess_fn  = load_fn(dp_cfg.path, dp_cfg.preprocess_fn_name)
        self.postprocess_fn = load_fn(dp_cfg.path, dp_cfg.postprocess_fn_name)

        # Load pairwise functions if pairwise_v1 is enabled
        if self.pairwise_v1:
            pairwise_preprocess_fn_name = dp_cfg.get("pairwise_preprocess_fn_name", "construct_deepseek_grm_inputs_pairwise")
            pairwise_postprocess_fn_name = dp_cfg.get("pairwise_postprocess_fn_name", "convert_deepseek_grm_pairwise_output_to_comparison")
            self.pairwise_preprocess_fn = load_fn(dp_cfg.path, pairwise_preprocess_fn_name)
            self.pairwise_postprocess_fn = load_fn(dp_cfg.path, pairwise_postprocess_fn_name)
            logger.info(f"Loaded pairwise functions: {pairwise_preprocess_fn_name}, {pairwise_postprocess_fn_name}")

        # Check if two_stage_grm mode is enabled
        # Two-stage mode: P(principle|question) * P(judge|question, principle, prediction)
        self.two_stage_grm = self.config.get("two_stage_grm", False)
        if self.two_stage_grm:
            # Load two-stage specific functions
            self.construct_principles_fn = load_fn(dp_cfg.path, "construct_principles_only_input")
            self.construct_judge_fn = load_fn(dp_cfg.path, "construct_judge_with_prefix_input")
            self.extract_principles_fn = load_fn(dp_cfg.path, "extract_principles_from_output")
            logger.info("Two-stage GRM mode enabled: P(principle|question) * P(judge|question, principle, prediction)")


        if self.model_type == "generative":
            assert self.preprocess_fn is not None and self.postprocess_fn is not None, (
                "generative reward model must have preprocess_fn and postprocess_fn"
            )

        # 2. build reward model device mesh
        infer_tp = self.config.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"reward model world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        reward_model_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        # Save device mesh for pairwise computation
        self._rm_device_mesh = reward_model_device_mesh

        is_collect = reward_model_device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            "reward_model", dp_rank=reward_model_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        # 3. init trainer and reward model random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = reward_model_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build reward model
        log_gpu_memory_usage("Before building sglang reward model", logger=logger)
        self.reward_model = get_reward_model_class(reward_model_config.name)(
            config=reward_model_config, model_config=model_config, device_mesh=reward_model_device_mesh
        )
        log_gpu_memory_usage("After building sglang reward model", logger=logger)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._build_reward_model()

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _preprocess_reward_inputs(self, data: DataProto):
        src_tokenizer = self.src_tokenizer
        tokenizer = self.tokenizer
        rm_inputs = []

        # Store original questions and responses for failure case logging
        self._original_questions = []
        self._original_responses = []

        for i in range(len(data)):
            data_item = data[i]

            # get rollout question
            if "extra_infos" in data_item.non_tensor_batch and "question" in data_item.non_tensor_batch["extra_infos"]:
                rollout_question = data_item.non_tensor_batch["extra_infos"]["question"]
            else:
                # use prompt_str as a substitute for question
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                # CRITICAL FIX: Convert tensor to Python int to avoid incorrect slicing
                valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                # IMPORTANT: Convert tensor to list before decoding to avoid garbled text
                rollout_question = src_tokenizer.decode(valid_prompt_ids.tolist(), skip_special_tokens=True)

            # get rollout response
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            # CRITICAL FIX: Convert tensor to Python int to avoid incorrect slicing
            valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            # IMPORTANT: Convert tensor to list before decoding to avoid garbled text
            rollout_response = src_tokenizer.decode(valid_response_ids.tolist(), skip_special_tokens=True)

            # Store for failure case logging
            self._original_questions.append(rollout_question)
            self._original_responses.append(rollout_response)

            # get ground truth
            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)

            if self.model_type == "discriminative":
                if self._do_switch_chat_template:
                    chats = [
                        {"role": "user", "content": rollout_question},
                        {"role": "assistant", "content": rollout_response},
                    ]
                    rm_input = tokenizer.apply_chat_template(chats, tokenize=True)
                else:
                    non_pad_indices = torch.nonzero(data_item.batch["attention_mask"], as_tuple=True)[0]
                    start_idx, end_idx = non_pad_indices[0], non_pad_indices[-1]
                    rm_input = data_item.batch["input_ids"][start_idx : end_idx + 1].tolist()
            else:
                assert self.preprocess_fn is not None, "generative reward model must have preprocess_fn"

                # Get principals from data if available (for Single-Stage GRM)
                principals = data_item.non_tensor_batch.get("principals", None)

                # Build kwargs for preprocess_fn
                preprocess_kwargs = {
                    "rollout_question": rollout_question,
                    "rollout_response": rollout_response,
                    "ground_truth": ground_truth,
                }
                # Add principals if available (for Single-Stage GRM with pre-generated principles)
                if principals is not None:
                    preprocess_kwargs["principals"] = principals

                input_str = self.preprocess_fn(**preprocess_kwargs)

                # For DeepSeek-GRM: the preprocess_fn already constructs the complete prompt
                # IMPORTANT: For vLLM compatibility with DeepSeek-GRM, we need to pass string prompts
                # instead of token IDs. vLLM has issues when using prompt_token_ids with DeepSeek V2 MoE.
                if "DeepSeek-GRM" in self.model_config.path:
                    # Return the string directly, let vLLM tokenize it
                    rm_input = input_str  # String, not token IDs!

                    # Debug: log the first few samples
                    import os
                    debug_log = "/workspace/qingnan/verl/examples/tmp/reward_model_debug.log"
                    if not os.path.exists(debug_log):
                        try:
                            with open(debug_log, 'w', encoding='utf-8') as f:
                                f.write("=== DeepSeek-GRM String Input Debug ===\n\n")
                                f.write(f"Input string length: {len(input_str)}\n")
                                f.write(f"Input string (first 15000 chars): {input_str[:15000]}\n\n")
                                f.write(f"Passing string directly to vLLM (not token IDs)\n")
                        except Exception as e:
                            pass
                else:
                    # For other generative reward models, use chat template
                    chats = [{"role": "user", "content": input_str}]
                    rm_input = tokenizer.apply_chat_template(chats, add_generation_prompt=True, tokenize=True)

            rm_inputs.append(rm_input)

        return rm_inputs

    def _postprocess_reward_outputs(self, data: DataProto, output: list[float] | list[list[int]] | list[str]):
        # Initialize metadata collection
        output_lengths_chars = []
        output_lengths_tokens = []
        extraction_methods = []
        failure_cases = []  # Store failure cases for logging

        # Define which extraction methods should be logged
        METHODS_TO_LOG = [
            "failed_return_0",
            "exception_return_0",
            "fallback_score_colon",
            "fallback_last_number",
        ]

        if self.model_type == "discriminative":
            scores = torch.tensor(output)
        else:
            assert self.postprocess_fn is not None, "generative reward model must have postprocess_fn"

            # Check if output is already text (detokenize=True) or token IDs (detokenize=False)
            if output and isinstance(output[0], str):
                # Already text strings from vLLM with detokenize=True
                output_text = output
            else:
                # Token IDs that need decoding
                # DEBUG: Log first output token IDs
                if output and len(output) > 0:
                    import os
                    debug_file = "/workspace/qingnan/verl/examples/tmp/token_ids_debug.log"
                    os.makedirs(os.path.dirname(debug_file), exist_ok=True)
                    with open(debug_file, 'a') as f:
                        f.write(f"\n=== Token IDs Debug ===\n")
                        f.write(f"First output token IDs: {output[0][:100]}\n")
                        f.write(f"Length: {len(output[0])}\n")
                        # Check for repeated token
                        if len(output[0]) > 10:
                            unique_tokens = set(output[0])
                            f.write(f"Unique tokens count: {len(unique_tokens)}\n")
                            f.write(f"Unique tokens: {list(unique_tokens)[:20]}\n")

                output_text = [self.tokenizer.decode(o) for o in output]

            # Get token counts from reward model if available
            if hasattr(self.reward_model, '_last_token_counts') and self.reward_model._last_token_counts:
                output_lengths_tokens = self.reward_model._last_token_counts
                self.reward_model._last_token_counts = None  # Clear after use
            else:
                # Fallback: estimate from text length
                output_lengths_tokens = []

            # Check if postprocess_fn supports return_metadata parameter
            import inspect
            sig = inspect.signature(self.postprocess_fn)
            supports_metadata = 'return_metadata' in sig.parameters

            # postprocess genrm responses to scores
            if supports_metadata:
                # Call with return_metadata=True to get detailed statistics
                results = [self.postprocess_fn(o, return_metadata=True) for o in output_text]
                scores = [r["score"] if isinstance(r, dict) else r for r in results]

                # Collect metadata if available
                for idx, r in enumerate(results):
                    if isinstance(r, dict):
                        output_lengths_chars.append(r.get("output_length", len(output_text[idx])))
                        method = r.get("extraction_method", "unknown")
                        extraction_methods.append(method)

                        # Collect failure cases for logging
                        if method in METHODS_TO_LOG:
                            # Get original question and response from preprocessing step
                            question = self._original_questions[idx] if hasattr(self, '_original_questions') and idx < len(self._original_questions) else "N/A"
                            response = self._original_responses[idx] if hasattr(self, '_original_responses') and idx < len(self._original_responses) else "N/A"

                            failure_cases.append({
                                "index": idx,
                                "extraction_method": method,
                                "score": r["score"],
                                "question": question,
                                "response": response,
                                "rm_output": output_text[idx],
                                "output_length": r.get("output_length", len(output_text[idx])),
                            })
                    else:
                        output_lengths_chars.append(len(output_text[idx]) if output_text else 0)
                        extraction_methods.append("no_metadata")
            else:
                # Legacy behavior: just get scores
                scores = [self.postprocess_fn(o) for o in output_text]
                # Estimate output lengths from text
                output_lengths_chars = [len(o) for o in output_text]
                extraction_methods = ["legacy_no_metadata"] * len(output_text)

            # ========== Handle invalid responses (score is None) ==========
            import logging
            logger = logging.getLogger(__name__)

            valid_mask = [s is not None for s in scores]
            num_invalid = sum(1 for v in valid_mask if not v)

            if num_invalid > 0:
                logger.warning(f"Found {num_invalid}/{len(scores)} responses with invalid reward scores (None)")

                # Replace None scores with 0.0 (neutral score for 1-10 scale)
                # These will be filtered later in ray_trainer based on invalid_mask
                scores_cleaned = [s if s is not None else 0.0 for s in scores]

                # Store invalid mask for later filtering in ray_trainer
                import numpy as np
                self._invalid_mask = np.array([not v for v in valid_mask], dtype=bool)

                scores = scores_cleaned
            else:
                # No invalid scores
                self._invalid_mask = None
            # ========== End of invalid handling logic ==========

            # ========== PENALTY FOR SELF-EVALUATION (Reward Hacking Prevention) ==========
            # Check if original responses contain self-grading patterns and penalize them
            # This is a CODE-LEVEL enforcement (not relying on reward model to follow instructions)
            SELF_EVAL_PATTERNS = [
                # Chinese patterns
                "特点总结：", "评分（满分10分）：", "综合评分：", "总评：", "总分：",
                "评分：", "得分：", "打分：", "分数：",
                # English patterns
                "Score (on 10-point scale):", "Tone & Style Summary:", "Overall Rating:",
                "Overall Score:", "Final Score:", "Total Score:",
                # Score patterns like "9.8/10", "9.9/10" etc at the end of response
                "/10 —", "/10—", "/10（", "/10 (",
            ]
            SCORE_PATTERN_REGEX = r'\b[89]\.\d/10\b|\b10/10\b|\b9\.[5-9]/10\b'

            PENALTY_SCORE = 3.0  # Assign low score for self-evaluation responses

            if hasattr(self, '_original_responses') and self._original_responses:
                import re
                num_penalized = 0
                for idx, response in enumerate(self._original_responses):
                    if idx >= len(scores):
                        break

                    # Check last 2000 characters of response for self-evaluation patterns
                    response_tail = response[-2000:] if len(response) > 2000 else response

                    is_self_eval = False
                    matched_pattern = None

                    # Check string patterns
                    for pattern in SELF_EVAL_PATTERNS:
                        if pattern in response_tail:
                            is_self_eval = True
                            matched_pattern = pattern
                            break

                    # Check regex pattern for scores like "9.8/10"
                    if not is_self_eval:
                        if re.search(SCORE_PATTERN_REGEX, response_tail):
                            is_self_eval = True
                            matched_pattern = "score_pattern_regex"

                    # Apply penalty
                    if is_self_eval:
                        original_score = scores[idx]
                        scores[idx] = PENALTY_SCORE
                        num_penalized += 1

                        # Log first few penalized cases
                        if num_penalized <= 3:
                            logger.warning(
                                f"[SELF-EVAL PENALTY] Sample {idx}: "
                                f"Detected pattern '{matched_pattern}', "
                                f"score {original_score:.1f} -> {PENALTY_SCORE:.1f}"
                            )

                if num_penalized > 0:
                    logger.warning(
                        f"[SELF-EVAL PENALTY] Penalized {num_penalized}/{len(scores)} responses "
                        f"for containing self-evaluation patterns (score -> {PENALTY_SCORE})"
                    )
            # ========== End of self-evaluation penalty logic ==========

            scores = torch.tensor(scores)

        token_level_scores = self._expand_to_token_level(data, scores)

        # Store metadata for later aggregation
        # We'll add this to the instance to be collected by compute_rm_score
        self._last_output_metadata = {
            "output_lengths_chars": output_lengths_chars,
            "output_lengths_tokens": output_lengths_tokens,
            "extraction_methods": extraction_methods,
            "failure_cases": failure_cases,
        }

        return token_level_scores

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward_model"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        data = data.to("cpu")

        # If pairwise_v1 is enabled, use pairwise comparison
        # IMPORTANT: Pairwise mode requires ALL responses from the same prompt to be together
        # But dispatch/chunk may split them across workers, so we need special handling
        if self.pairwise_v1:
            token_level_scores = self._compute_pairwise_scores_with_gather(data)
        elif self.two_stage_grm:
            # Two-stage GRM: P(principle|question) * P(judge|question, principle, prediction)
            token_level_scores = self._compute_two_stage_grm_scores(data)
        else:
            # Standard pointwise scoring
            rm_data = self._preprocess_reward_inputs(data)
            output = self.reward_model.compute_reward(rm_data)
            token_level_scores = self._postprocess_reward_outputs(data, output)

        # Collect metadata from the postprocessing step
        # Separate batch-level data (arrays) from scalar statistics
        metadata = {}  # For batch-level arrays (goes to non_tensors)
        meta_info = {}  # For scalar statistics and dicts (goes to meta_info)

        # IMPORTANT: Add invalid_mask to non_tensor_batch if some samples have invalid scores
        # This will be used by ray_trainer to filter out invalid samples after union
        import numpy as np
        if hasattr(self, '_invalid_mask') and self._invalid_mask is not None:
            # Some samples have invalid scores (were set to 0.0)
            metadata["invalid_mask"] = self._invalid_mask
            meta_info["num_invalid_scores"] = int(np.sum(self._invalid_mask))
            self._invalid_mask = None  # Clear after use
        else:
            # No invalid scores - create a False mask for all samples
            batch_size = data.batch.batch_size[0]
            metadata["invalid_mask"] = np.zeros(batch_size, dtype=bool)
            meta_info["num_invalid_scores"] = 0

        if hasattr(self, '_last_output_metadata') and self._last_output_metadata:
            output_lengths_chars = self._last_output_metadata.get("output_lengths_chars", [])
            output_lengths_tokens = self._last_output_metadata.get("output_lengths_tokens", [])
            extraction_methods = self._last_output_metadata.get("extraction_methods", [])
            failure_cases = self._last_output_metadata.get("failure_cases", [])

            import numpy as np

            # Store character-based lengths for reference (batch-level array)
            if output_lengths_chars:
                metadata["rm_output_lengths_chars"] = output_lengths_chars

            # Prefer token-based lengths for metrics (more accurate)
            output_lengths = output_lengths_tokens if output_lengths_tokens else output_lengths_chars

            if output_lengths:
                # Batch-level arrays go to metadata (non_tensors)
                metadata["rm_output_lengths"] = output_lengths
                metadata["rm_extraction_methods"] = extraction_methods

                # Scalar statistics go to meta_info (not non_tensors)
                meta_info["rm_output_length_min"] = int(np.min(output_lengths))
                meta_info["rm_output_length_mean"] = float(np.mean(output_lengths))
                meta_info["rm_output_length_max"] = int(np.max(output_lengths))

                # Dictionary data also goes to meta_info
                from collections import Counter
                method_counts = Counter(extraction_methods)
                meta_info["rm_extraction_method_counts"] = dict(method_counts)

            # Add failure cases for logging (goes to meta_info, not metadata)
            # failure_cases is not a per-sample array, but a list of failure details
            if failure_cases:
                meta_info["rm_failure_cases"] = failure_cases

            # Clear the metadata for next batch
            self._last_output_metadata = None

        # Note that this is only the scores, may not be the final rewards used to train RL
        output = DataProto.from_dict(
            tensors={"rm_scores": token_level_scores},
            non_tensors=metadata if metadata else {},
            meta_info=meta_info if meta_info else {}
        )
        return output

    def _compute_two_stage_grm_scores(self, data: DataProto):
        """
        Compute rewards using two-stage GRM:
        P(principle|question) * P(judge|question, principle, prediction)
        
        Stage 1: Generate evaluation principles based on question only (shared across responses)
        Stage 2: Evaluate each response using the shared principles as generation prefix
        
        Args:
            data: DataProto containing batch of responses
            
        Returns:
            token_level_scores: Tensor of shape (batch_size, response_length)
        """
        from collections import defaultdict
        import numpy as np
        
        batch_size = data.batch.batch_size[0]
        src_tokenizer = self.src_tokenizer
        
        # Get data processor config for template settings
        dp_cfg = self.data_processor_config
        strip_think_tag = dp_cfg.get("strip_think_tag", True)
        template_version = dp_cfg.get("template_version", "v6")
        
        # ========== Step 1: Extract questions and responses, group by question ==========
        questions = []
        responses = []
        question_to_indices = defaultdict(list)  # question -> list of indices
        
        # Store for failure case logging
        self._original_questions = []
        self._original_responses = []
        
        for i in range(batch_size):
            data_item = data[i]
            
            # Get question
            if "extra_infos" in data_item.non_tensor_batch and "question" in data_item.non_tensor_batch["extra_infos"]:
                question = data_item.non_tensor_batch["extra_infos"]["question"]
            else:
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                question = src_tokenizer.decode(valid_prompt_ids.tolist(), skip_special_tokens=True)
            
            # Get response
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            response = src_tokenizer.decode(valid_response_ids.tolist(), skip_special_tokens=True)
            
            questions.append(question)
            responses.append(response)
            question_to_indices[question].append(i)
            
            # Store for logging
            self._original_questions.append(question)
            self._original_responses.append(response)
        
        # ========== Step 2: Generate principles for each unique question (Stage 1) ==========
        unique_questions = list(question_to_indices.keys())
        
        # Construct stage 1 inputs (principles generation)
        stage1_inputs = []
        for q in unique_questions:
            stage1_input = self.construct_principles_fn(
                rollout_question=q,
                template_version=template_version
            )
            stage1_inputs.append(stage1_input)
        
        # Call reward model for stage 1
        logger.info(f"Two-stage GRM Stage 1: Generating principles for {len(unique_questions)} unique questions")
        stage1_outputs = self.reward_model.compute_reward(stage1_inputs)
        
        # Process stage 1 outputs to extract principles
        if stage1_outputs and isinstance(stage1_outputs[0], str):
            stage1_texts = stage1_outputs
        else:
            stage1_texts = [self.tokenizer.decode(o) for o in stage1_outputs]
        
        question_to_principles = {}
        for q, output_text in zip(unique_questions, stage1_texts):
            principles = self.extract_principles_fn(output_text)
            question_to_principles[q] = principles
        
        # Debug logging for stage 1
        log_file = "/data/qingnan/verl_1214/examples/tmp/two_stage_grm_debug.log"
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    count = content.count('=== STAGE1 BATCH #')
            else:
                count = 0
            
            if count < 3:
                with open(log_file, 'a', encoding='utf-8') as f:
                    import time
                    f.write(f"\n{'='*80}\n")
                    f.write(f"=== STAGE1 BATCH #{count + 1} ===\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Batch size: {batch_size}\n")
                    f.write(f"Unique questions: {len(unique_questions)}\n")
                    f.write(f"Responses per question: {[len(question_to_indices[q]) for q in unique_questions]}\n\n")
                    
                    for i, (q, principles) in enumerate(question_to_principles.items()):
                        if i < 2:  # Only log first 2 questions
                            f.write(f"--- Question {i+1} (first 500 chars) ---\n{q[:500]}\n\n")
                            f.write(f"--- Generated Principles ---\n{principles}\n\n")
        except Exception as e:
            pass
        
        # ========== Step 3: Evaluate each response with shared principles (Stage 2) ==========
        # Prepare stage 2 inputs in order
        stage2_inputs = []
        for i in range(batch_size):
            question = questions[i]
            response = responses[i]
            principles = question_to_principles[question]
            
            stage2_input = self.construct_judge_fn(
                rollout_question=question,
                rollout_response=response,
                principles_prefix=principles,
                strip_think_tag=strip_think_tag,
                template_version=template_version
            )
            stage2_inputs.append(stage2_input)
        
        # Call reward model for stage 2
        logger.info(f"Two-stage GRM Stage 2: Evaluating {batch_size} responses with shared principles")
        stage2_outputs = self.reward_model.compute_reward(stage2_inputs)
        
        # ========== Step 4: Process outputs to get scores ==========
        token_level_scores = self._postprocess_reward_outputs(data, stage2_outputs)
        
        return token_level_scores

    def _compute_pairwise_scores_with_gather(self, data: DataProto):
        """
        Compute pairwise scores with a memory-efficient strategy.

        Since batch may be split across workers, we have two strategies:
        1. If dp_size is small (<=2), use all_gather (acceptable memory cost)
        2. If dp_size is large, use local-only pairwise comparison within each worker
        """
        from torch.distributed.device_mesh import DeviceMesh

        # Get the reward model device mesh
        if not hasattr(self, '_rm_device_mesh') or self._rm_device_mesh is None:
            # If no device mesh or only one worker, just use the regular function
            if self.world_size == 1:
                return self._compute_pairwise_scores(data)
            logger.warning("No device mesh found, falling back to local pairwise computation")
            return self._compute_pairwise_scores(data)

        # Get the data parallelism process group
        dp_mesh = self._rm_device_mesh["dp"]
        dp_size = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()

        if dp_size == 1:
            # No data parallelism, just compute locally
            return self._compute_pairwise_scores(data)

        # Memory-efficient strategy based on dp_size
        if dp_size <= 2:
            # Small number of workers: use all_gather (memory overhead is acceptable)
            logger.info(f"Pairwise mode: using all_gather with dp_size={dp_size}")
            from verl.protocol import all_gather_data_proto
            local_batch_size = data.batch.batch_size[0]
            all_gather_data_proto(data, process_group=dp_mesh.get_group())
            full_scores = self._compute_pairwise_scores(data)
            start_idx = dp_rank * local_batch_size
            end_idx = start_idx + local_batch_size
            return full_scores[start_idx:end_idx]
        else:
            # Large number of workers: use local-only comparison (no all_gather)
            # This saves memory but may have incomplete groups
            logger.warning(
                f"Pairwise mode: dp_size={dp_size} is large, using local-only comparison. "
                f"Some groups may be incomplete. Consider using tensor_model_parallel_size > 1 "
                f"to reduce dp_size."
            )
            return self._compute_pairwise_scores(data)

    def _compute_pairwise_scores(self, data: DataProto):
        """
        Compute rewards using pairwise comparison (pairwise_v1 mode).

        For each group of N responses:
        1. Randomly select one response as baseline
        2. Compare all other responses with the baseline
        3. Assign rewards:
           - baseline: -1
           - better than baseline: +1
           - worse than baseline: -1

        Args:
            data: DataProto containing batch of responses

        Returns:
            token_level_scores: Tensor of shape (batch_size, response_length)
        """
        import random
        import numpy as np
        from collections import defaultdict

        batch_size = data.batch.batch_size[0]

        # Extract prompt indices to group responses
        # In GRPO, responses from the same prompt are grouped together
        # verl uses "uid" field to track which prompt each response belongs to
        if "uid" in data.non_tensor_batch:
            indices = data.non_tensor_batch["uid"]
        elif "index" in data.non_tensor_batch:
            indices = data.non_tensor_batch["index"]
        else:
            # If no uid/index is provided, assume each response is from a different prompt
            logger.warning("No 'uid' or 'index' found in data, treating each response as a separate group")
            indices = list(range(batch_size))

        # Debug: log the first few times to understand grouping
        import os
        debug_file = "/workspace/qingnan/verl/examples/tmp/pairwise_grouping_debug.log"
        os.makedirs(os.path.dirname(debug_file), exist_ok=True)

        try:
            if not os.path.exists(debug_file):
                with open(debug_file, 'w') as f:
                    f.write(f"=== Pairwise Grouping Debug ===\n")
                    f.write(f"Batch size: {batch_size}\n")
                    f.write(f"Indices type: {type(indices)}\n")
                    f.write(f"Indices shape: {indices.shape if hasattr(indices, 'shape') else len(indices)}\n")
                    f.write(f"Unique indices: {np.unique(indices)}\n")
                    f.write(f"Indices (first 20): {indices[:min(20, len(indices))]}\n")
                    from collections import Counter
                    counter = Counter(indices.tolist() if hasattr(indices, 'tolist') else indices)
                    f.write(f"Group sizes: {dict(counter)}\n")
        except Exception as e:
            pass

        # Group data items by prompt index
        groups = defaultdict(list)
        for i in range(batch_size):
            groups[indices[i]].append(i)

        # Initialize scores with zeros
        scores = torch.zeros(batch_size)

        # Process each group
        for group_idx, item_indices in groups.items():
            if len(item_indices) < 2:
                # If only one response in the group, give it a neutral score (0)
                logger.warning(f"Group {group_idx} has only 1 response, assigning neutral score")
                scores[item_indices[0]] = 0.0
                continue

            # Randomly select a baseline response
            baseline_idx = random.choice(item_indices)
            baseline_item = data[baseline_idx]

            # Get baseline question and response
            if "extra_infos" in baseline_item.non_tensor_batch and "question" in baseline_item.non_tensor_batch["extra_infos"]:
                baseline_question = baseline_item.non_tensor_batch["extra_infos"]["question"]
            else:
                prompt_ids = baseline_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = int(baseline_item.batch["attention_mask"][:prompt_length].sum().item())
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                baseline_question = self.src_tokenizer.decode(valid_prompt_ids.tolist(), skip_special_tokens=True)

            response_ids = baseline_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = int(baseline_item.batch["attention_mask"][-response_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            baseline_response = self.src_tokenizer.decode(valid_response_ids.tolist(), skip_special_tokens=True)

            # Baseline gets reward -1
            scores[baseline_idx] = -1.0

            # Compare all other responses with baseline
            pairwise_inputs = []
            compare_indices = []

            for idx in item_indices:
                if idx == baseline_idx:
                    continue

                # Get current response
                current_item = data[idx]
                response_ids = current_item.batch["responses"]
                response_length = response_ids.shape[-1]
                valid_response_length = int(current_item.batch["attention_mask"][-response_length:].sum().item())
                valid_response_ids = response_ids[:valid_response_length]
                current_response = self.src_tokenizer.decode(valid_response_ids.tolist(), skip_special_tokens=True)

                # Construct pairwise comparison input
                # response1: current response, response2: baseline response
                pairwise_input = self.pairwise_preprocess_fn(
                    rollout_question=baseline_question,
                    response1=current_response,
                    response2=baseline_response
                )

                pairwise_inputs.append(pairwise_input)
                compare_indices.append(idx)

            # Call reward model for pairwise comparisons
            if pairwise_inputs:
                outputs = self.reward_model.compute_reward(pairwise_inputs)

                # Check if outputs are already text strings or token IDs
                if outputs and isinstance(outputs[0], str):
                    output_text = outputs
                else:
                    output_text = [self.tokenizer.decode(o) for o in outputs]

                # Process each comparison result
                for i, (output, idx) in enumerate(zip(output_text, compare_indices)):
                    comparison_result = self.pairwise_postprocess_fn(output)
                    # comparison_result: 1 if current > baseline, -1 if current <= baseline
                    scores[idx] = float(comparison_result)

        # Expand to token level
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]

        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]

        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (batch_size,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (batch_size, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # Select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores
