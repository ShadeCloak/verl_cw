#!/usr/bin/env python3
"""
Analyze the length distribution of DeepWriting-20K dataset
Compare original solution length vs. cleaned content length (after removing thinking process)
"""

import argparse
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
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


def collect_length_statistics(dataset) -> Tuple[List[int], List[int]]:
    """
    Collect length statistics for original and cleaned solutions.

    Returns:
        Tuple of (original_lengths, cleaned_lengths)
    """
    original_lengths = []
    cleaned_lengths = []

    print(f"Processing {len(dataset)} examples...")

    for i, example in enumerate(dataset):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(dataset)} examples")

        solution = example.get('solution', '')

        # Original length
        original_length = len(solution)
        original_lengths.append(original_length)

        # Cleaned length (after removing thinking process)
        cleaned_content = extract_final_content(solution)
        cleaned_length = len(cleaned_content)
        cleaned_lengths.append(cleaned_length)

    print(f"Completed processing {len(dataset)} examples")

    return original_lengths, cleaned_lengths


def plot_length_distributions(original_lengths: List[int],
                              cleaned_lengths: List[int],
                              output_path: str = None):
    """
    Plot the length distributions of original and cleaned content.
    """
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Convert to numpy arrays for easier manipulation
    original = np.array(original_lengths)
    cleaned = np.array(cleaned_lengths)

    # Calculate statistics
    print("\n=== Length Statistics ===")
    print(f"Original Content:")
    print(f"  Mean: {original.mean():.2f} chars")
    print(f"  Median: {np.median(original):.2f} chars")
    print(f"  Std: {original.std():.2f} chars")
    print(f"  Min: {original.min()} chars")
    print(f"  Max: {original.max()} chars")
    print(f"  95th percentile: {np.percentile(original, 95):.2f} chars")

    print(f"\nCleaned Content (after removing <think>):")
    print(f"  Mean: {cleaned.mean():.2f} chars")
    print(f"  Median: {np.median(cleaned):.2f} chars")
    print(f"  Std: {cleaned.std():.2f} chars")
    print(f"  Min: {cleaned.min()} chars")
    print(f"  Max: {cleaned.max()} chars")
    print(f"  95th percentile: {np.percentile(cleaned, 95):.2f} chars")

    # Calculate reduction
    reduction = original - cleaned
    print(f"\nReduction (thinking process removed):")
    print(f"  Mean reduction: {reduction.mean():.2f} chars")
    print(f"  Median reduction: {np.median(reduction):.2f} chars")
    print(f"  Examples with thinking process: {np.sum(reduction > 0)} ({np.sum(reduction > 0) / len(reduction) * 100:.2f}%)")

    # Subplot 1: Overlapping histograms (full range)
    ax1 = axes[0, 0]
    bins = np.linspace(0, max(original.max(), cleaned.max()), 100)
    ax1.hist(original, bins=bins, alpha=0.6, label='Original', color='blue', edgecolor='black')
    ax1.hist(cleaned, bins=bins, alpha=0.6, label='Cleaned (no <think>)', color='orange', edgecolor='black')
    ax1.set_xlabel('Length (characters)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Length Distribution (Full Range)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Zoomed in to 95th percentile
    ax2 = axes[0, 1]
    max_len_95 = max(np.percentile(original, 95), np.percentile(cleaned, 95))
    bins_zoomed = np.linspace(0, max_len_95, 100)
    ax2.hist(original, bins=bins_zoomed, alpha=0.6, label='Original', color='blue', edgecolor='black')
    ax2.hist(cleaned, bins=bins_zoomed, alpha=0.6, label='Cleaned (no <think>)', color='orange', edgecolor='black')
    ax2.set_xlabel('Length (characters)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Length Distribution (0-95th percentile)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Box plot comparison
    ax3 = axes[1, 0]
    box_data = [original, cleaned]
    bp = ax3.boxplot(box_data, labels=['Original', 'Cleaned\n(no <think>)'],
                     patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightyellow')
    ax3.set_ylabel('Length (characters)', fontsize=12)
    ax3.set_title('Length Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Subplot 4: Reduction distribution
    ax4 = axes[1, 1]
    ax4.hist(reduction, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(reduction.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {reduction.mean():.0f} chars')
    ax4.axvline(np.median(reduction), color='purple', linestyle='--', linewidth=2,
                label=f'Median: {np.median(reduction):.0f} chars')
    ax4.set_xlabel('Length Reduction (characters)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Thinking Process Length Distribution', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze length distribution of DeepWriting-20K dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (for testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/qingnan/verl/examples/creative/data_preprocess/length_distribution.png",
        help="Output path for the plot"
    )

    args = parser.parse_args()

    print("Loading DeepWriting-20K dataset...")
    dataset = load_dataset("m-a-p/DeepWriting-20K", trust_remote_code=True)
    train_dataset = dataset['train']

    if args.max_samples:
        print(f"Limiting to {args.max_samples} samples for testing")
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))

    print(f"Loaded {len(train_dataset)} examples")

    # Collect length statistics
    original_lengths, cleaned_lengths = collect_length_statistics(train_dataset)

    # Plot distributions
    plot_length_distributions(original_lengths, cleaned_lengths, args.output)

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
