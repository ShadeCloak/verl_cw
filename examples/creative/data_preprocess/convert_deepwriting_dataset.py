#!/usr/bin/env python3
"""
Convert m-a-p/DeepWriting-20K dataset to VERL-compatible format
"""

import argparse
import json
import os
import re
from typing import Dict, Any, List

import datasets
from datasets import load_dataset


def extract_final_content(solution_text: str) -> str:
    """
    Extract the final writing content from solution text.
    The solution often contains <think>...</think> sections that should be removed.
    """
    if not solution_text:
        return ""

    # Remove <think>...</think> sections using regex
    think_pattern = r'<think>.*?</think>\s*'
    cleaned_text = re.sub(think_pattern, '', solution_text, flags=re.DOTALL)

    # Strip extra whitespace
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def is_creative_writing_task(prompt_content: str, ability: str, domain_info: Dict) -> bool:
    """
    Determine if this is a creative writing task based on prompt and metadata.
    """
    # Check ability field
    if ability and ability.lower() in ['writing', 'creative_writing', 'narrative']:
        return True

    # Check domain information
    if domain_info and 'category' in domain_info and domain_info['category']:
        category = domain_info['category'].lower()
        if 'creative' in category or 'narrative' in category or 'artistic' in category:
            return True

    # Check prompt content for creative writing keywords
    creative_keywords = [
        '写一篇文章', '写作', '散文', '小说', '故事', '诗歌',
        'write a story', 'write an article', 'creative writing',
        'narrative', 'fiction', 'essay', 'prose'
    ]

    if prompt_content:
        prompt_lower = prompt_content.lower()
        for keyword in creative_keywords:
            if keyword in prompt_lower:
                return True

    return False


def convert_deepwriting_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single DeepWriting-20K example to VERL format.
    """
    # Extract basic information
    prompt = example.get('prompt', [])
    solution = example.get('solution', '')
    ability = example.get('ability', 'writing')
    original_data_source = example.get('data_source', 'open-ended')  # Keep original for metadata
    extra_info = example.get('extra_info', {})

    # Get prompt content for analysis
    prompt_content = ""
    if prompt and len(prompt) > 0 and 'content' in prompt[0]:
        prompt_content = prompt[0]['content']

    # Extract final content from solution
    final_content = extract_final_content(solution)

    # Determine if this is creative writing
    is_creative = is_creative_writing_task(prompt_content, ability, extra_info.get('domain', {}))

    # Create the VERL-compatible data structure
    verl_example = {
        'data_source': 'DeepWriting-20K',
        'prompt': prompt,  # Already in correct format
        'ability': 'creative_writing' if is_creative else ability,
        'extra_info': {
            'index': extra_info.get('index', ''),
            'chinese': extra_info.get('chinese', False),
            'domain': extra_info.get('domain', {}),
            'original_data_source': original_data_source,
            'has_thinking_process': '<think>' in solution
        }
    }

    # Set up reward model configuration for creative writing
    # if is_creative:
        # For creative writing, we'll use the final content as ground truth
        # and set up for reward model evaluation
    verl_example['reward_model'] = {
        'style': 'model',
        'ground_truth': final_content,
        'evaluation_criteria': ['creativity', 'coherence', 'fluency', 'task_completion'],
    }
    # else:
    #     # For non-creative writing tasks, keep original reward model setup
    #     original_reward_model = example.get('reward_model', {})
    #     verl_example['reward_model'] = {
    #         'style': original_reward_model.get('style', 'rule'),
    #         'ground_truth': original_reward_model.get('ground_truth', final_content)
    #     }

    return verl_example


def process_dataset_split(dataset_split, split_name: str) -> datasets.Dataset:
    """
    Process a dataset split and convert all examples.
    """
    print(f"Processing {split_name} split with {len(dataset_split)} examples...")

    converted_examples = []

    for i, example in enumerate(dataset_split):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(dataset_split)} examples")

        try:
            converted_example = convert_deepwriting_example(example)
            converted_examples.append(converted_example)
        except Exception as e:
            print(f"  Warning: Failed to convert example {i}: {e}")
            continue

    print(f"Successfully converted {len(converted_examples)} examples")

    # Create a new dataset from converted examples
    return datasets.Dataset.from_list(converted_examples)


def save_converted_dataset(dataset, output_dir: str, split_name: str):
    """
    Save the converted dataset in parquet format.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save as parquet
    parquet_path = os.path.join(output_dir, f"{split_name}.parquet")
    dataset.to_parquet(parquet_path)
    print(f"Saved {split_name} dataset to {parquet_path}")

    # Save a sample as JSON for inspection
    if len(dataset) > 0:
        sample_path = os.path.join(output_dir, f"{split_name}_sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(dataset[0], f, indent=2, ensure_ascii=False)
        print(f"Saved sample to {sample_path}")


def create_train_test_split(dataset, test_size: float = 0.01):
    """
    Create train/test split from the dataset.
    """
    total_size = len(dataset)
    test_split_size = int(total_size * test_size)

    # Shuffle and split
    shuffled_dataset = dataset.shuffle(seed=42)

    test_dataset = shuffled_dataset.select(range(test_split_size))
    train_dataset = shuffled_dataset.select(range(test_split_size, total_size))

    print(f"Created train split: {len(train_dataset)} examples")
    print(f"Created test split: {len(test_dataset)} examples")

    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description="Convert DeepWriting-20K dataset to VERL format")
    parser.add_argument("--output_dir", default="/workspace/qingnan/verl/examples/creative/data",
                       help="Output directory for converted dataset")
    parser.add_argument("--test_size", type=float, default=0.01,
                       help="Fraction of data to use for test set")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")

    args = parser.parse_args()

    print("Loading DeepWriting-20K dataset...")
    try:
        # Load the dataset
        dataset = load_dataset("m-a-p/DeepWriting-20K", trust_remote_code=True)
        original_train = dataset['train']

        if args.max_samples:
            print(f"Limiting to {args.max_samples} samples for testing")
            original_train = original_train.select(range(min(args.max_samples, len(original_train))))

        print(f"Loaded dataset with {len(original_train)} examples")

        # Convert the dataset
        converted_dataset = process_dataset_split(original_train, "full")

        # Create train/test split
        train_dataset, test_dataset = create_train_test_split(converted_dataset, args.test_size)

        # Save both splits
        save_converted_dataset(train_dataset, args.output_dir, "train")
        save_converted_dataset(test_dataset, args.output_dir, "test")

        # Create a summary
        summary = {
            "total_examples": len(converted_dataset),
            "train_examples": len(train_dataset),
            "test_examples": len(test_dataset),
            "test_size_ratio": args.test_size,
            "creative_writing_examples": sum(1 for ex in converted_dataset if ex['ability'] == 'creative_writing'),
            "chinese_examples": sum(1 for ex in converted_dataset if ex['extra_info'].get('chinese', False)),
            "examples_with_thinking": sum(1 for ex in converted_dataset if ex['extra_info'].get('has_thinking_process', False))
        }

        summary_path = os.path.join(args.output_dir, "conversion_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nConversion completed successfully!")
        print(f"Summary saved to {summary_path}")
        print(f"Output directory: {args.output_dir}")

        # Print summary
        print("\n=== Conversion Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()