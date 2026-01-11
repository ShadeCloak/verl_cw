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
Process functions for DeepSeek-GRM generative reward model.

DeepSeek-GRM is a generative reward model following the "principle → critique → score" pipeline.
It evaluates responses based on comprehensive criteria and outputs scores in \boxed{} format.

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


# Standard prompt template following official DeepSeek-GRM format
DEEPSEEK_GRM_PROMPT_TEMPLATE_v0 = """You are a skilled expert at scoring responses. You should evaluate the given response based on the judging criteria.
Given the context of the conversation (the User's query) and the response from the Assistant, you need to refer to the [General Evaluation Criteria] to score the response. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score.
The score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

#### Evaluation Criteria ####
1. Instruction Adherence:
   - Fully Adhered (9-10 points): The response fully complies with all instructions and requirements of the question.
   - Partially Adhered (6-8 points): The response meets most of the instructions but has some omissions or misunderstandings.
   - Basically Adhered (3-5 points): The response meets some instructions, but the main requirements are not fulfilled.
   - Not Adhered (1-2 points): The response does not meet any instructions.
   Example: If the question requires three examples and the response provides only one, it falls under "Partially Adhered."

2. Usefulness:
   - Highly Useful (9-10 points): The response provides comprehensive and accurate information, fully addressing the issue.
   - Useful but Incomplete (6-8 points): The response provides some useful information, but lacks details or accuracy.
   - Limited Usefulness (3-5 points): The response offers little useful information, with most content being irrelevant or incorrect.
   - Useless or Incorrect (1-2 points): The response is completely irrelevant or incorrect.
   Example: If there are factual errors in the response but the overall direction is correct, it falls under "Useful but Incomplete."

3. Level of Detail:
   - Very Detailed (9-10 points): The response includes ample details covering all aspects of the issue.
   - Detailed but Slightly Lacking (6-8 points): The response is fairly detailed but misses some important details.
   - Basically Detailed (3-5 points): The response provides some details but is not thorough enough overall.
   - Not Detailed (1-2 points): The response is very brief and lacks necessary details.
   Example: If the response provides only a simple conclusion without an explanation, it falls under "Not Detailed."

4. Relevance:
   - Highly Relevant (9-10 points): The response is highly relevant to the question, with information closely aligned with the topic.
   - Generally Relevant (6-8 points): The response is generally relevant but includes some unnecessary information.
   - Partially Relevant (3-5 points): The response has a lot of content that deviates from the topic.
   - Not Relevant (1-2 points): The response is completely irrelevant.
   Example: If the response strays from the topic but still provides some relevant information, it falls under "Partially Relevant."

#### Conversation Context ####
User: {question}

#### Response to be Scored ####
[The Begin of Response]
{response}
[The End of Response]

#### Output Format Requirements ####
You MUST output exactly in this format:
Specific Criteria: <Other potential criteria specific to the query and the context, and the weights of each criteria>.
Analysis: <Analyze the response based on the given and specific criteria>.
Score: \\boxed{{X}} where X is an integer from 1 to 10.

IMPORTANT: The final score MUST be in the exact format \\boxed{{X}}. For example, if the score is 8, you must write: Score: \\boxed{{8}}"""


DEEPSEEK_GRM_PROMPT_TEMPLATE = """You are a skilled expert at scoring responses. You should evaluate the given response based on the judging criteria.
Given the context of the conversation (the User's query) and the response from the Assistant, you need to refer to the [General Evaluation Criteria] to score the response. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score.
The score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

#### Evaluation Criteria ####
1. Instruction Compliance & Consistency:
Evaluate whether the response strictly follows the user’s instructions and whether the type, format, and scope of the output precisely match the request. Also check for unnecessary additions or off-topic content.
[Key Focus Areas]: Does the output strictly match the requested content type (e.g., story only, summary only)? Does it include unsolicited extras (e.g., unnecessary explanations, analyses, or comments after the main result)? Is there any mismatch between request and output (e.g., outline instead of summary, pseudocode instead of code)? Is the response internally consistent and free of contradictions?
   
   - Highly Accurate (9-10 points): The response strictly follows the instructions regarding content type, format, and scope; no extra or missing material; perfectly consistent logic; precisely fulfills the prompt without extraneous detail.
   - Generally Accurate (6-8 points): Core requirements met; minor unsolicited additions or slight format deviations; small logical slips.
   - Partially Accurate (3-5 points): Content type or key elements mismatch; noticeable extra or omitted sections (e.g., lengthy explanations, analyses, or advice after the main result); contradictions present.
   - Not Accurate (1-2 points): Completely wrong content type or ignores instructions; severe contradictions or incoherent logic.
   Example: A user asks for a short story and receives a story plus two paragraphs of analysis → score 3-5 (Partially Accurate).

2. Usefulness:
   - Highly Useful (9-10 points): The response provides comprehensive and accurate information, fully addressing the issue.
   - Useful but Incomplete (6-8 points): The response provides some useful information, but lacks details or accuracy.
   - Limited Usefulness (3-5 points): The response offers little useful information, with most content being irrelevant or incorrect.
   - Useless or Incorrect (1-2 points): The response is completely irrelevant or incorrect.
   Example: If there are factual errors in the response, it falls under "Useless or Incorrect."

3. Level of Detail:
   - Very Detailed (9-10 points): The response includes ample details covering all aspects of the issue.
   - Detailed but Slightly Lacking (6-8 points): The response is fairly detailed but misses some important details.
   - Basically Detailed (3-5 points): The response provides some details but is not thorough enough overall.
   - Not Detailed (1-2 points): The response is very brief and lacks necessary details.
   Example: If the response provides only a superficial answer lacking depth, it falls under "Basically Detailed."

4. Relevance:
   - Highly Relevant (9-10 points): The response is highly relevant to the question, with information closely aligned with the topic.
   - Generally Relevant (6-8 points): The response is generally relevant but includes some unnecessary information.
   - Partially Relevant (3-5 points): The response has a lot of content that deviates from the topic.
   - Not Relevant (1-2 points): The response is completely irrelevant.
   Example: If the response strays from the topic, it falls under "Not Relevant."

5. Factual Accuracy  
Assesses whether the information in the response is accurate and reliable, free from fabrication, fiction, or error.

   - Fully Accurate (9-10 points): All facts, data, and citations are correct; no hallucinations or fabricated content.  
   - Basically Accurate (6-8 points): Most information is correct; only minor factual errors or imprecise statements.  
   - Partially Accurate (3-5 points): Noticeable factual errors or invented content, yet the core information is directionally correct.  
   - Severely Inaccurate (1-2 points): Contains extensive false information, hallucinations, or seriously misleading statements.  
   Example: Inventing non-existent elements, incorrect historical dates, or fabricated research data all count as "Severely Inaccurate."

#### Conversation Context ####
User: {question}

#### Response to be Scored ####
[The Begin of Response]
{response}
[The End of Response]

#### Output Format Requirements ####
You MUST output exactly in this format:
Specific Criteria: <Other potential criteria specific to the query and the context, and the weights of each criteria>.
Analysis: <Analyze the response based on the given and specific criteria>.
Score: \\boxed{{X}} where X is an integer from 1 to 10.

IMPORTANT: The final score MUST be in the exact format \\boxed{{X}}. For example, if the score is 8, you must write: Score: \\boxed{{8}}"""



DEEPSEEK_GRM_PROMPT_TEMPLATE_v6 = """You are a skilled expert at scoring responses. You should evaluate the given response based on judging criteria.

Given the User's query and the Assistant's response, you need to:
1. **First, generate specific evaluation principles/criteria** tailored to this particular query
2. Analyze the response based on these criteria
3. Provide an overall comprehensive score (1-10)

Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

#### Conversation Context ####
User: {question}

#### Response to be Scored ####
[The Begin of Response]
{response}
[The End of Response]

#### Output Format Requirements ####
You MUST output exactly in this format:
Evaluation Principles: <List the specific evaluation principles/criteria for this query, including their relative weights>
Analysis: <Analyze the response based on the stated principles>
Score: \\boxed{{X}} where X is an integer from 1 to 10.

IMPORTANT: The final score MUST be in the exact format \\boxed{{X}}. For example, if the score is 8, you must write: Score: \\boxed{{8}}"""


# v8 版本：v6 + few-shot（预定义 Evaluation Principles 示例）
DEEPSEEK_GRM_PROMPT_TEMPLATE_v8 = """You are a skilled expert at scoring responses. You should evaluate the given response based on judging criteria.

Given the User's query and the Assistant's response, you need to:
1. **First, generate specific evaluation principles/criteria** tailored to this particular query
2. Analyze the response based on these criteria
3. Provide an overall comprehensive score (1-10)

Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

#### Conversation Context ####
User: {question}

#### Response to be Scored ####
[The Begin of Response]
{response}
[The End of Response]

#### Output Format Requirements ####
You MUST output exactly in this format:
Evaluation Principles: <List the specific evaluation principles/criteria for this query, including their relative weights>
Analysis: <Analyze the response based on the stated principles>
Score: \\boxed{{X}} where X is an integer from 1 to 10.

IMPORTANT: The final score MUST be in the exact format \\boxed{{X}}. For example, if the score is 8, you must write: Score: \\boxed{{8}}"""

# v8 的 few-shot assistant 示例
DEEPSEEK_GRM_FEWSHOT_ASSISTANT_v8 = """Evaluation Principles:
1. Factual Accuracy (Weight: 30%): The response must be factually correct and reliable, ensuring all data, facts, and logic are free from hallucinations, fabrications, or errors.
2. Instruction Compliance & Consistency (Weight: 20%): The response must strictly adhere to the user's constraints regarding format, content type, and scope, while maintaining internal logic without contradictory or unsolicited information.
"""


DEEPSEEK_GRM_PROMPT_TEMPLATE_PAIRWISE = """You are a skilled expert at scoring responses. You should evaluate the given response based on the judging criteria.
Given the context of the conversation (the User's query) and the response from the Assistant, you need to refer to the [General Evaluation Criteria] to score the response. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score.
The score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

#### Evaluation Criteria ####
1. Instruction Adherence:
   - Fully Adhered (9-10 points): The response fully complies with all instructions and requirements of the question.
   - Partially Adhered (6-8 points): The response meets most of the instructions but has some omissions or misunderstandings.
   - Basically Adhered (3-5 points): The response meets some instructions, but the main requirements are not fulfilled.
   - Not Adhered (1-2 points): The response does not meet any instructions.
   Example: If the question requires three examples and the response provides only one, it falls under "Partially Adhered."

2. Usefulness:
   - Highly Useful (9-10 points): The response provides comprehensive and accurate information, fully addressing the issue.
   - Useful but Incomplete (6-8 points): The response provides some useful information, but lacks details or accuracy.
   - Limited Usefulness (3-5 points): The response offers little useful information, with most content being irrelevant or incorrect.
   - Useless or Incorrect (1-2 points): The response is completely irrelevant or incorrect.
   Example: If there are factual errors in the response but the overall direction is correct, it falls under "Useful but Incomplete."

3. Level of Detail:
   - Very Detailed (9-10 points): The response includes ample details covering all aspects of the issue.
   - Detailed but Slightly Lacking (6-8 points): The response is fairly detailed but misses some important details.
   - Basically Detailed (3-5 points): The response provides some details but is not thorough enough overall.
   - Not Detailed (1-2 points): The response is very brief and lacks necessary details.
   Example: If the response provides only a simple conclusion without an explanation, it falls under "Not Detailed."

4. Relevance:
   - Highly Relevant (9-10 points): The response is highly relevant to the question, with information closely aligned with the topic.
   - Generally Relevant (6-8 points): The response is generally relevant but includes some unnecessary information.
   - Partially Relevant (3-5 points): The response has a lot of content that deviates from the topic.
   - Not Relevant (1-2 points): The response is completely irrelevant.
   Example: If the response strays from the topic but still provides some relevant information, it falls under "Partially Relevant."

#### Conversation Context ####
User: {question}

#### Response to be Scored ####
[The Begin of Response 1]
{response1}
[The End of Response 1]

[The Begin of Response 2]
{response2}
[The End of Response 2]

#### Output Format Requirements ####
Output with three lines:
Specific Criteria: <Other potential criteria specific to the query and the context, and the weights of each criteria>.
Analysis: <Analyze the response based on the given and specific criteria>.
Scores: <the overall comprehensive score of all resposnes in order, seperate by comma in the boxed, e.g., \\boxed{{x, x}} if there exists 2 responeses>."""


def construct_deepseek_grm_inputs(rollout_question: str, rollout_response: str, ground_truth=None,
                                  strip_think_tag: bool = True, template_version: str = "v1") -> str:
    """
    Construct input prompt for DeepSeek-GRM model following the official format.

    This function creates a prompt that follows DeepSeek-GRM's standard "principle → critique → score"
    pipeline. The model will:
    1. Generate specific evaluation criteria for this query
    2. Analyze the response based on criteria
    3. Output a score from 1-10 in \\boxed{} format

    Args:
        rollout_question: The user's question/prompt
        rollout_response: The model's generated response to evaluate
        ground_truth: Optional ground truth answer (not used for general evaluation)
        strip_think_tag: Whether to strip content before </think> tag. Default: True
        template_version: Which prompt template to use ("v1" or "v0"). Default: "v1"

    Returns:
        Formatted input string for the DeepSeek-GRM reward model
    """
    # Truncate if response is too long to fit within context window
    MAX_RESPONSE_CHARS = 30000  # Roughly 7500 tokens, adjust based on your max_model_len
    if len(rollout_response) > MAX_RESPONSE_CHARS:
        # Keep both beginning and end as they often contain important context and conclusions
        truncation_msg = f"\n... [truncated {len(rollout_response) - MAX_RESPONSE_CHARS} characters] ...\n"
        half_len = MAX_RESPONSE_CHARS // 2
        rollout_response = rollout_response[:half_len] + truncation_msg + rollout_response[-half_len:]

    # Similarly truncate question if needed (though less common)
    MAX_QUESTION_CHARS = 10000
    if len(rollout_question) > MAX_QUESTION_CHARS:
        truncation_msg = f"\n... [truncated {len(rollout_question) - MAX_QUESTION_CHARS} characters] ...\n"
        half_len = MAX_QUESTION_CHARS // 2
        rollout_question = rollout_question[:half_len] + truncation_msg + rollout_question[-half_len:]

    # Strip </think> tag if requested (controlled by hyperparameter)
    if strip_think_tag and '</think>' in rollout_response:
        rollout_response = rollout_response.split('</think>')[-1].strip()

    # Select prompt template based on version (controlled by hyperparameter)
    # Also determine if we need to use few-shot mode
    use_fewshot = False
    fewshot_assistant = None

    if template_version == "v0":
        template = DEEPSEEK_GRM_PROMPT_TEMPLATE_v0
    elif template_version == "v1":
        template = DEEPSEEK_GRM_PROMPT_TEMPLATE
    elif template_version == "v6":
        template = DEEPSEEK_GRM_PROMPT_TEMPLATE_v6
    elif template_version == "v8":
        template = DEEPSEEK_GRM_PROMPT_TEMPLATE_v8
        use_fewshot = True
        fewshot_assistant = DEEPSEEK_GRM_FEWSHOT_ASSISTANT_v8
    else:
        # Default to v1 if unknown version specified
        template = DEEPSEEK_GRM_PROMPT_TEMPLATE

    # Format the prompt using the selected template
    prompt = template.format(
        question=rollout_question,
        response=rollout_response
    )

    # Apply chat template using the official DeepSeek-GRM tokenizer
    # This ensures the correct format with <｜User｜> and <｜Assistant｜> markers
    tokenizer = _get_deepseek_grm_tokenizer()

    # For v8, use few-shot mode by adding assistant's partial response
    if use_fewshot and fewshot_assistant is not None:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": fewshot_assistant}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,           # Return string, not token IDs
        add_generation_prompt=True  # Add <｜Assistant｜> at the end
    )

    # Simple logging: just append to a single log file
    log_file = "/data/qingnan/verl_1214/examples/tmp/deepseek_grm_debug.log"
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # Count existing entries
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                count = content.count('=== INPUT #')
        else:
            count = 0

        # Only log first 10 inputs
        if count < 10:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"=== INPUT #{count + 1} ===\n")
                f.write(f"{'='*80}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Question Length: {len(rollout_question)}\n")
                f.write(f"Response Length: {len(rollout_response)}\n")
                f.write(f"Original Prompt Length: {len(prompt)}\n")
                f.write(f"Formatted Prompt Length: {len(formatted_prompt)}\n\n")
                f.write(f"--- Question ---\n{rollout_question}\n\n")
                f.write(f"--- Response ---\n{rollout_response}\n\n")
                f.write(f"--- Original Prompt ---\n{prompt}\n\n")
                f.write(f"--- Formatted Prompt (with chat template) ---\n{formatted_prompt}\n\n")
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
    Specific Criteria: ...
    Analysis: ...
    Score: \\boxed{8}

    This function extracts the score from \\boxed{} and normalizes it to [0, 1] range.

    Args:
        output: Text output from the DeepSeek-GRM model
        return_metadata: If True, return dict with score and extraction metadata

    Returns:
        If return_metadata=False: Normalized reward score in [0, 1] range
        If return_metadata=True: Dict with keys:
            - score: The extracted score
            - extraction_method: Which method was used to extract the score
            - output_length: Length of the output text in characters
    """
    # Simple logging: append output to the same log file
    log_file = "/data/qingnan/verl_1214/examples/tmp/deepseek_grm_debug.log"

    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                count = content.count('=== OUTPUT #')
        else:
            count = 0

        # Only log first 10 outputs
        if count < 10:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"=== OUTPUT #{count + 1} ===\n")
                f.write(f"{'='*80}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output Length: {len(output)}\n\n")
                f.write(f"--- DeepSeek-GRM Evaluation ---\n{output}\n\n")
    except Exception as e:
        pass  # Don't break if logging fails

    # Track metadata
    output_length = len(output)
    extraction_method = "unknown"

    try:
        # Use the official extraction function
        scores = extract_last_floats(output)

        if scores and len(scores) > 0:
            # Take the first score (we only evaluate one response at a time)
            score = scores[0]

            # DeepSeek-GRM outputs scores from 1 to 10
            # Return the original score without normalization
            # GRPO algorithm will perform its own group-wise z-score normalization,
            # so we don't need to normalize here. Keeping original 1-10 range
            # makes the scores more interpretable in logs.

            # Clip to [1, 10] range to handle any edge cases
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
            # If no score found in \boxed{}, try comprehensive fallback patterns
            # The model may output scores in various formats, we try to extract them

            # Pattern 1: "Score: X" or "score: X" (English)
            # Pattern 2: "评分: X" or "综合评分: X" (Chinese)
            # Pattern 3: Numbers at the end of the text (last resort)
            # Pattern 4: Numbers in parentheses like "(5 points)" or "(5分)"

            fallback_patterns = [
                ("fallback_score_colon", r'(?:Score|score|SCORE)[:\s]+(?:is\s+)?(\d+\.?\d*)'),  # Score: 8 or Score is 8
                ("fallback_chinese", r'(?:评分|综合评分|得分)[:\s：]+(\d+\.?\d*)'),  # Chinese score patterns
                ("fallback_points", r'\((\d+\.?\d*)\s*(?:points|分|pts)\)'),  # (5 points) or (5分)
                ("fallback_rating", r'(?:rating|Rating)[:\s]+(\d+\.?\d*)'),  # Rating: 8
                ("fallback_overall", r'(?:total|Total|overall|Overall)[:\s]+(\d+\.?\d*)'),  # Total/Overall: 8
            ]

            for method_name, pattern in fallback_patterns:
                matches = list(re.finditer(pattern, output))
                if matches:
                    # Take the last match (most likely to be the final score)
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

            # Last resort: look for standalone numbers at the end of the text
            # This is risky but better than returning a default
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

            # If still no score found, return None to indicate invalid response
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
        return None  # Return None to indicate invalid response on error


# ============================================================================
# Pairwise Comparison Functions (for pairwise_v1 mode)
# ============================================================================

def construct_deepseek_grm_inputs_pairwise(rollout_question: str, response1: str, response2: str, ground_truth=None) -> str:
    """
    Construct pairwise comparison input for DeepSeek-GRM model.

    This function creates a prompt for comparing two responses using DeepSeek-GRM.
    The model will evaluate both responses and output scores in \\boxed{score1, score2} format.

    Args:
        rollout_question: The user's question/prompt
        response1: The first response to compare
        response2: The second response to compare
        ground_truth: Optional ground truth answer (not used for general evaluation)

    Returns:
        Formatted input string for the DeepSeek-GRM reward model
    """
    # Truncate if responses are too long to fit within context window
    MAX_RESPONSE_CHARS = 15000  # Shorter limit for pairwise to fit both responses

    if len(response1) > MAX_RESPONSE_CHARS:
        truncation_msg = f"\n... [truncated {len(response1) - MAX_RESPONSE_CHARS} characters] ...\n"
        half_len = MAX_RESPONSE_CHARS // 2
        response1 = response1[:half_len] + truncation_msg + response1[-half_len:]

    if len(response2) > MAX_RESPONSE_CHARS:
        truncation_msg = f"\n... [truncated {len(response2) - MAX_RESPONSE_CHARS} characters] ...\n"
        half_len = MAX_RESPONSE_CHARS // 2
        response2 = response2[:half_len] + truncation_msg + response2[-half_len:]

    # Similarly truncate question if needed
    MAX_QUESTION_CHARS = 10000
    if len(rollout_question) > MAX_QUESTION_CHARS:
        truncation_msg = f"\n... [truncated {len(rollout_question) - MAX_QUESTION_CHARS} characters] ...\n"
        half_len = MAX_QUESTION_CHARS // 2
        rollout_question = rollout_question[:half_len] + truncation_msg + rollout_question[-half_len:]

    # Format the prompt using the pairwise template
    prompt = DEEPSEEK_GRM_PROMPT_TEMPLATE_PAIRWISE.format(
        question=rollout_question,
        response1=response1,
        response2=response2
    )

    # Apply chat template using the official DeepSeek-GRM tokenizer
    tokenizer = _get_deepseek_grm_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,           # Return string, not token IDs
        add_generation_prompt=True  # Add <｜Assistant｜> at the end
    )

    # Logging for debugging
    log_file = "/data/qingnan/verl_1214/examples/tmp/deepseek_grm_pairwise_debug.log"
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                count = content.count('=== PAIRWISE INPUT #')
        else:
            count = 0

        # Only log first 5 pairwise inputs
        if count < 5:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"=== PAIRWISE INPUT #{count + 1} ===\n")
                f.write(f"{'='*80}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Question Length: {len(rollout_question)}\n")
                f.write(f"Response1 Length: {len(response1)}\n")
                f.write(f"Response2 Length: {len(response2)}\n")
                f.write(f"Formatted Prompt Length: {len(formatted_prompt)}\n\n")
                f.write(f"--- Question ---\n{rollout_question}\n\n")
                f.write(f"--- Response 1 ---\n{response1}\n\n")
                f.write(f"--- Response 2 ---\n{response2}\n\n")
                f.write(f"--- Formatted Prompt (with chat template) ---\n{formatted_prompt}\n\n")
    except Exception as e:
        pass  # Don't break if logging fails

    return formatted_prompt


def convert_deepseek_grm_pairwise_output_to_comparison(output: str) -> int:
    """
    Convert DeepSeek-GRM pairwise comparison output to a binary result.

    DeepSeek-GRM outputs scores in the format:
    Specific Criteria: ...
    Analysis: ...
    Scores: \\boxed{score1, score2}

    This function extracts the two scores and returns:
    - 1 if response1 is better than response2
    - -1 if response2 is better than response1

    Args:
        output: Text output from the DeepSeek-GRM model

    Returns:
        1 if response1 is better, -1 otherwise
    """
    # Logging
    log_file = "/data/qingnan/verl_1214/examples/tmp/deepseek_grm_pairwise_debug.log"

    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                count = content.count('=== PAIRWISE OUTPUT #')
        else:
            count = 0

        # Only log first 5 outputs
        if count < 5:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"=== PAIRWISE OUTPUT #{count + 1} ===\n")
                f.write(f"{'='*80}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output Length: {len(output)}\n\n")
                f.write(f"--- DeepSeek-GRM Pairwise Evaluation ---\n{output}\n\n")
    except Exception as e:
        pass  # Don't break if logging fails

    try:
        # Extract scores using the same extraction function
        scores = extract_last_floats(output)

        if scores and len(scores) >= 2:
            score1, score2 = scores[0], scores[1]

            # Clip scores to [1, 10] range
            score1 = max(1.0, min(10.0, score1))
            score2 = max(1.0, min(10.0, score2))

            # Compare: if response1 is better, return 1; otherwise return -1
            return 1 if score1 > score2 else -1
        else:
            # If we can't extract both scores, try to find them in the text
            print(f"Warning: Could not extract two scores from pairwise output. Found {len(scores)} scores.")
            print(f"Output (first 500 chars): {output[:500]}")
            # Default to -1 (response2 is better) in case of error
            return -1

    except Exception as e:
        print(f"Error processing DeepSeek-GRM pairwise output: {e}")
        print(f"Output (first 2000 chars): {output[:2000]}")
        return -1  # Default to -1 in case of error
