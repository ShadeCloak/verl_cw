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
Process functions for DeepSeek-GRM - Single Stage Version (优化版)

基于 /verl_cw/examples/creative/deepseek_grm_process_fn.py 的 Two-Stage GRM 修改而来。

优化思路:
- **原来 (Two-Stage GRM)**: 每个 response 调用 2 次 RM
  - Stage 1 (construct_principles_only_input): 生成 Evaluation Principles
  - Stage 2 (construct_judge_with_prefix_input): 使用 Principles 进行评分
- **现在 (Single-Stage GRM)**: 每个 response 只调用 1 次 RM
  - Principals 已预生成并存储在数据的 'principals' 列中
  - 直接使用预生成的 Principles 进行评分 (相当于只执行 Stage 2)

实现对比:
- Two-Stage: construct_judge_with_prefix_input(question, response, principles_from_stage1)
- Single-Stage: construct_deepseek_grm_single_stage_input(question, response, principals_from_data)

训练数据要求:
- 必须包含 'principals' 列，存储预生成的 Evaluation Principles
- 使用 generate_principals_for_dataset.py 批量生成

Official documentation: https://huggingface.co/BBQGOD/DeepSeek-GRM-16B
"""

import re
import os
import time

# Lazy load tokenizer at module level (only loaded once when first needed)
_deepseek_grm_tokenizer = None

def _get_deepseek_grm_tokenizer():
    """
    Get the DeepSeek-GRM tokenizer (lazy loaded and cached).
    This ensures the tokenizer is only loaded once, even if the function is called multiple times.
    """
    global _deepseek_grm_tokenizer
    if _deepseek_grm_tokenizer is None:
        from transformers import AutoTokenizer
        _deepseek_grm_tokenizer = AutoTokenizer.from_pretrained(
            "/data/qingnan/model/DeepSeek-GRM-16B",
            trust_remote_code=True
        )
    return _deepseek_grm_tokenizer


# 使用与 Two-Stage 相同的第二阶段模板
# 来自 /verl_cw/examples/creative/deepseek_grm_process_fn.py 的 DEEPSEEK_GRM_JUDGE_WITH_PRINCIPLES_TEMPLATE
# 这是 Two-Stage GRM 的 Stage 2 模板，Single-Stage 直接使用它（跳过 Stage 1）
DEEPSEEK_GRM_JUDGE_WITH_PRINCIPLES_TEMPLATE = r"""You are a skilled expert at scoring responses. Based on the given evaluation principles, analyze the response and provide a comprehensive score.

Scoring Guidelines:
- The score is a number with one decimal place between 1.0 and 10.0
- Score 9.0-10.0: Exceptional response that fully meets all criteria with outstanding quality
- Score 7.0-9.0: Good response that meets most criteria with minor areas for improvement
- Score 5.0-7.0: Adequate response that meets basic requirements but has noticeable weaknesses
- Score 3.0-5.0: Below average response with significant issues or missing key elements
- Score below 3.0: Poor response that fails to meet most criteria or contains major errors
- PENALTY FOR SELF-EVALUATION: Check the end of the response carefully. If the response includes any self-grading, self-analysis, or meta-summaries (specifically containing phrases such as "特点总结：", "评分（满分10分）：", "Score (on 10-point scale):", "Tone & Style Summary:", or explicit scores like "9.9/10"), this is considered a critical violation. You MUST assign a final score strictly lower than 5.0, regardless of the quality of the rest of the content.

#### User Query ####
{question}

#### Response to be Scored ####
[The Begin of Response]
{response}
[The End of Response]

#### Evaluation Principles (Pre-defined) ####
{principle}

#### Output Format Requirements ####
Based on the above evaluation principles, you MUST output exactly in this format:

Analysis:
- **[Criterion 1 Name]**: <Detailed analysis of performance on this criterion, explaining strengths and weaknesses>. Score: X.X/10.0
- **[Criterion 2 Name]**: <Detailed analysis of performance on this criterion, explaining strengths and weaknesses>. Score: X.X/10.0
- **[Criterion 3 Name]**: <Detailed analysis of performance on this criterion, explaining strengths and weaknesses>. Score: X.X/10.0
- **[Criterion 4 Name]**: <Detailed analysis of performance on this criterion, explaining strengths and weaknesses>. Score: X.X/10.0
- **[Criterion 5 Name]**: <Detailed analysis of performance on this criterion, explaining strengths and weaknesses>. Score: X.X/10.0

Conclusion: <A comprehensive summary of your analysis, highlighting main strengths and weaknesses>

Final Score (Weighted Average): <Show the calculation: weight1×score1 + weight2×score2 + ... = final_score>

Score: \boxed{{X.X}}

CRITICAL REQUIREMENTS:
1. In "Analysis", provide detailed analysis for each criterion from the given principles
2. Each criterion MUST be scored out of 10.0 (format: "Score: X.X/10.0")
3. The final score MUST be the weighted average of all criterion scores based on the given weights
4. Show your weighted average calculation explicitly before the boxed score
6. The final boxed score must have one decimal place
7. PENALTY FOR SELF-EVALUATION: Check the end of the response carefully. If the response includes any self-grading, self-analysis, or meta-summaries (specifically containing phrases such as "特点总结：", "评分（满分10分）：", "Score (on 10-point scale):", "Tone & Style Summary:", or explicit scores like "9.9/10"), this is considered a critical violation. You MUST assign a final score strictly lower than 5.0, regardless of the quality of the rest of the content."""


def construct_deepseek_grm_single_stage_input(
    rollout_question: str,
    rollout_response: str,
    principals: str,
    ground_truth=None,
    strip_think_tag: bool = True,
    template_version: str = "single_stage"
) -> str:
    """
    构建单次调用的输入 prompt (使用预生成的 principals)

    基于 deepseek_grm_process_fn.py 的 construct_judge_with_prefix_input() 修改而来。
    主要区别: principals 参数来自数据的 'principals' 列 (预生成),
              而非 Two-Stage 中的 Stage 1 动态生成。

    这个函数直接使用数据中预生成的 principals，一次性完成评分。
    相比 Two-Stage 模式，节省了一次 reward model 调用 (跳过 Stage 1)。

    Args:
        rollout_question: 用户的问题/prompt
        rollout_response: 模型生成的回复 (需要评分)
        principals: 预生成的 Evaluation Principles (从数据的 principals 列读取)
        ground_truth: 可选的标准答案 (本函数中不使用)
        strip_think_tag: 是否去除 </think> 标签前的内容. Default: True
        template_version: 模板版本 (本函数中不使用，保留以兼容配置)

    Returns:
        格式化的输入字符串，供 DeepSeek-GRM reward model 使用
    """
    # 截断过长的 response
    MAX_RESPONSE_CHARS = 30000  # 约 7500 tokens
    if len(rollout_response) > MAX_RESPONSE_CHARS:
        truncation_msg = f"\n... [truncated {len(rollout_response) - MAX_RESPONSE_CHARS} characters] ...\n"
        half_len = MAX_RESPONSE_CHARS // 2
        rollout_response = rollout_response[:half_len] + truncation_msg + rollout_response[-half_len:]

    # 截断过长的 question
    MAX_QUESTION_CHARS = 10000
    if len(rollout_question) > MAX_QUESTION_CHARS:
        truncation_msg = f"\n... [truncated {len(rollout_question) - MAX_QUESTION_CHARS} characters] ...\n"
        half_len = MAX_QUESTION_CHARS // 2
        rollout_question = rollout_question[:half_len] + truncation_msg + rollout_question[-half_len:]

    # 去除 </think> 标签 (如果需要)
    if strip_think_tag and '</think>' in rollout_response:
        rollout_response = rollout_response.split('</think>')[-1].strip()

    # 格式化 prompt (使用与 Two-Stage 相同的第二阶段模板)
    prompt = DEEPSEEK_GRM_JUDGE_WITH_PRINCIPLES_TEMPLATE.format(
        question=rollout_question,
        response=rollout_response,
        principle=principals  # 直接使用预生成的 principals (来自数据，而非 Stage 1 生成)
    )

    # 应用 chat template
    tokenizer = _get_deepseek_grm_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Debug logging (记录前 10 个输入)
    log_file = "/data/qingnan/verl_1214/examples/tmp/deepseek_grm_single_stage_debug.log"
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                count = content.count('=== SINGLE STAGE INPUT #')
        else:
            count = 0

        if count < 10:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"=== SINGLE STAGE INPUT #{count + 1} ===\n")
                f.write(f"{'='*80}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Question Length: {len(rollout_question)}\n")
                f.write(f"Response Length: {len(rollout_response)}\n")
                f.write(f"Principals Length: {len(principals)}\n")
                f.write(f"Formatted Prompt Length: {len(formatted_prompt)}\n\n")
                f.write(f"--- Question ---\n{rollout_question[:500]}...\n\n")
                f.write(f"--- Response ---\n{rollout_response[:500]}...\n\n")
                f.write(f"--- Principals ---\n{principals[:500]}...\n\n")
                f.write(f"--- Formatted Prompt (with chat template) ---\n{formatted_prompt[:1000]}...\n\n")
    except Exception as e:
        pass  # Don't break if logging fails

    return formatted_prompt


def extract_last_floats(text: str) -> list[float]:
    """
    Extract floats from the last \\boxed{} or [] in the text.

    This is the official extraction function from DeepSeek-GRM documentation.
    It handles formats like:
    - \\boxed{8}
    - \\boxed{8.5}
    - [8]
    - \\boxed{7, 8, 9} (for multiple responses, returns list)

    Args:
        text: Generated text from DeepSeek-GRM

    Returns:
        List of float scores found in the last boxed expression
    """
    pattern = re.compile(
        r'(?:\\{1,2}boxed\{|\[)'
        r'\s*([^\]\}]+?)\s*'
        r'(?:\}|\])'
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    last_content = matches[-1].group(1)
    parts = re.split(r'\s*,\s*', last_content.strip())
    floats = []
    for p in parts:
        try:
            floats.append(float(p))
        except ValueError:
            pass
    return floats


def convert_deepseek_grm_output_to_reward(output: str, return_metadata: bool = False) -> float | dict:
    """
    Convert DeepSeek-GRM text output to a numerical reward score.

    DeepSeek-GRM outputs scores in the format:
    Analysis: ...
    Conclusion: ...
    Final Score (Weighted Average): ...
    Score: \\boxed{8.5}

    This function extracts the score from \\boxed{} and returns it.

    Args:
        output: Text output from the DeepSeek-GRM model
        return_metadata: If True, return dict with score and extraction metadata

    Returns:
        If return_metadata=False: Score (1-10 range, not normalized)
        If return_metadata=True: Dict with keys:
            - score: The extracted score
            - extraction_method: Which method was used to extract the score
            - output_length: Length of the output text in characters
    """
    # Debug logging
    log_file = "/data/qingnan/verl_1214/examples/tmp/deepseek_grm_single_stage_debug.log"

    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                count = content.count('=== SINGLE STAGE OUTPUT #')
        else:
            count = 0

        if count < 10:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"=== SINGLE STAGE OUTPUT #{count + 1} ===\n")
                f.write(f"{'='*80}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output Length: {len(output)}\n\n")
                f.write(f"--- DeepSeek-GRM Evaluation ---\n{output}\n\n")
    except Exception as e:
        pass

    # Track metadata
    output_length = len(output)
    extraction_method = "unknown"

    try:
        # Use the official extraction function
        scores = extract_last_floats(output)

        if scores and len(scores) > 0:
            score = scores[0]

            # DeepSeek-GRM outputs scores from 1 to 10
            # Keep original 1-10 range (GRPO will do group-wise normalization)
            score = max(1.0, min(10.0, score))
            extraction_method = "primary_boxed"

            if return_metadata:
                return {
                    "score": score,
                    "extraction_method": extraction_method,
                    "output_length": output_length,
                }
            return score
        else:
            # Fallback patterns
            fallback_patterns = [
                ("fallback_score_colon", r'(?:Score|score|SCORE)[:\s]+(?:is\s+)?(\d+\.?\d*)'),
                ("fallback_chinese", r'(?:评分|综合评分|得分)[:\s：]+(\d+\.?\d*)'),
                ("fallback_points", r'\((\d+\.?\d*)\s*(?:points|分|pts)\)'),
                ("fallback_rating", r'(?:rating|Rating)[:\s]+(\d+\.?\d*)'),
                ("fallback_overall", r'(?:total|Total|overall|Overall)[:\s]+(\d+\.?\d*)'),
            ]

            for method_name, pattern in fallback_patterns:
                matches = list(re.finditer(pattern, output))
                if matches:
                    score = float(matches[-1].group(1))
                    score = max(1.0, min(10.0, score))
                    extraction_method = method_name
                    print(f"Warning: Score extracted from fallback pattern '{pattern}': {score}")

                    if return_metadata:
                        return {
                            "score": score,
                            "extraction_method": extraction_method,
                            "output_length": output_length,
                        }
                    return score

            # Last resort: look for numbers at the end
            last_500_chars = output[-500:]
            number_matches = list(re.finditer(r'\b([1-9]|10)(?:\.0)?\b', last_500_chars))
            if number_matches:
                score = float(number_matches[-1].group(1))
                score = max(1.0, min(10.0, score))
                extraction_method = "fallback_last_number"
                print(f"Warning: Score extracted from last number in text: {score}")

                if return_metadata:
                    return {
                        "score": score,
                        "extraction_method": extraction_method,
                        "output_length": output_length,
                    }
                return score

            # No score found
            print(f"Warning: Could not extract score from output. First 200 chars: {output[:200]}")
            extraction_method = "failed_return_none"

            if return_metadata:
                return {
                    "score": None,
                    "extraction_method": extraction_method,
                    "output_length": output_length,
                }
            return None

    except Exception as e:
        print(f"Error processing DeepSeek-GRM output: {e}")
        print(f"Output (first 2000 chars): {output[:2000]}")
        extraction_method = "exception_return_none"

        if return_metadata:
            return {
                "score": None,
                "extraction_method": extraction_method,
                "output_length": output_length,
            }
        return None
