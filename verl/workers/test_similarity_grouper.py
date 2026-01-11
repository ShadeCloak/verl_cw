#!/usr/bin/env python3
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
Test script for similarity grouper module.

This script tests the similarity grouping functionality without requiring
the full training setup.
"""

import numpy as np
import ray

from similarity_grouper import (
    SimilarityGrouperWorker,
    compute_group_distribution_per_prompt,
    compute_single_member_mask,
)


def test_basic_grouping():
    """Test basic similarity grouping functionality."""
    print("=" * 80)
    print("Test 1: Basic Similarity Grouping")
    print("=" * 80)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_gpus=1)

    # Create a single worker
    worker = SimilarityGrouperWorker.remote(model_name="BAAI/bge-m3")

    # Test data: 2 prompts, each with 8 responses
    # Prompt 0: 3 similar groups
    results_prompt_0 = [
        "The cat sat on the mat.",
        "A cat was sitting on a mat.",  # Similar to above
        "The dog ran in the park.",
        "A dog was running in a park.",  # Similar to above
        "Dogs play in parks.",  # Similar to above
        "Python is a programming language.",
        "Java is used for software development.",  # Similar to above
        "JavaScript runs in browsers.",  # Similar to above
    ]

    # Prompt 1: 2 similar groups
    results_prompt_1 = [
        "Machine learning is fascinating.",
        "Deep learning is a subset of ML.",
        "Neural networks are powerful.",
        "AI will change the world.",  # All similar
        "The weather is nice today.",
        "It's sunny and warm outside.",
        "Beautiful day for a walk.",
        "Perfect weather for outdoor activities.",  # All similar (different topic)
    ]

    results = results_prompt_0 + results_prompt_1
    uids = [0] * 8 + [1] * 8

    # Compute grouping
    print("\nComputing similarity groups...")
    result = ray.get(worker.compute_groups_for_batch.remote(
        results=results,
        uids=uids,
        k=8
    ))

    similarity_group_labels = result['similarity_group_labels']
    group_stats = result['group_stats']

    print(f"\nPrompt 0 groups:")
    for i, (text, label) in enumerate(zip(results_prompt_0, similarity_group_labels[:8])):
        print(f"  [{label}] {text[:50]}...")

    print(f"\nPrompt 1 groups:")
    for i, (text, label) in enumerate(zip(results_prompt_1, similarity_group_labels[8:])):
        print(f"  [{label}] {text[:50]}...")

    print(f"\nGroup Statistics:")
    print(f"  Number of prompts: {group_stats['n_prompts']}")
    print(f"  Groups per prompt: {group_stats['n_groups_per_prompt']}")
    print(f"  Silhouette scores: {[f'{s:.3f}' for s in group_stats['silhouette_scores']]}")

    # Test helper functions
    print("\n" + "=" * 80)
    print("Test 2: Helper Functions")
    print("=" * 80)

    # Test group distribution
    group_distribution = compute_group_distribution_per_prompt(
        similarity_group_labels, uids
    )
    print(f"\nGroup distribution (n_groups -> count):")
    for k in sorted(group_distribution.keys()):
        print(f"  {k} groups: {group_distribution[k]} prompts")

    # Test single-member mask
    single_member_mask = compute_single_member_mask(similarity_group_labels)
    n_single_members = single_member_mask.sum()
    print(f"\nSingle-member groups: {n_single_members} out of {len(similarity_group_labels)} responses")

    if n_single_members > 0:
        print("Single-member responses:")
        for i, (is_single, text) in enumerate(zip(single_member_mask, results)):
            if is_single:
                print(f"  [{similarity_group_labels[i]}] {text[:50]}...")

    ray.shutdown()
    print("\n✓ All tests passed!")


def test_edge_cases():
    """Test edge cases: identical responses, very different responses."""
    print("\n" + "=" * 80)
    print("Test 3: Edge Cases")
    print("=" * 80)

    if not ray.is_initialized():
        ray.init(num_gpus=1)

    worker = SimilarityGrouperWorker.remote(model_name="BAAI/bge-m3")

    # Case 1: All identical responses (should be 1 group)
    print("\nCase 1: All identical responses")
    identical_results = ["The same text."] * 8
    result = ray.get(worker.compute_groups_for_batch.remote(
        results=identical_results,
        uids=[0] * 8,
        k=8
    ))
    print(f"  Groups: {set(result['similarity_group_labels'])}")
    print(f"  Expected: 1 group (0_0)")
    print(f"  Silhouette: {result['group_stats']['silhouette_scores'][0]:.3f}")

    # Case 2: All very different responses (should be many groups)
    print("\nCase 2: Very different responses")
    different_results = [
        "Cats are mammals.",
        "Python programming language.",
        "Quantum physics theories.",
        "Ancient Roman history.",
        "Modern art movements.",
        "Climate change effects.",
        "Basketball game rules.",
        "Cooking Italian pasta.",
    ]
    result = ray.get(worker.compute_groups_for_batch.remote(
        results=different_results,
        uids=[1] * 8,
        k=8
    ))
    unique_groups = len(set(result['similarity_group_labels']))
    print(f"  Unique groups: {unique_groups}")
    print(f"  Silhouette: {result['group_stats']['silhouette_scores'][0]:.3f}")

    ray.shutdown()
    print("\n✓ Edge case tests passed!")


def test_advantage_computation():
    """Test that advantage computation works with similarity group labels."""
    print("\n" + "=" * 80)
    print("Test 4: Advantage Computation")
    print("=" * 80)

    import torch
    from verl.trainer.ppo import core_algos

    # Simulate a batch with 2 prompts, 8 responses each
    # Prompt 0: 3 groups [0_0, 0_0, 0_1, 0_1, 0_1, 0_2, 0_2, 0_2]
    # Prompt 1: 2 groups [1_0, 1_0, 1_0, 1_0, 1_1, 1_1, 1_1, 1_1]
    similarity_group_labels = [
        "0_0", "0_0", "0_1", "0_1", "0_1", "0_2", "0_2", "0_2",
        "1_0", "1_0", "1_0", "1_0", "1_1", "1_1", "1_1", "1_1",
    ]

    # Simulate rewards
    token_level_rewards = torch.tensor([
        [1.0, 0.0, 0.0],  # Group 0_0
        [1.5, 0.0, 0.0],  # Group 0_0
        [5.0, 0.0, 0.0],  # Group 0_1
        [5.5, 0.0, 0.0],  # Group 0_1
        [6.0, 0.0, 0.0],  # Group 0_1
        [2.0, 0.0, 0.0],  # Group 0_2
        [2.5, 0.0, 0.0],  # Group 0_2
        [3.0, 0.0, 0.0],  # Group 0_2
        [7.0, 0.0, 0.0],  # Group 1_0
        [7.5, 0.0, 0.0],  # Group 1_0
        [8.0, 0.0, 0.0],  # Group 1_0
        [8.5, 0.0, 0.0],  # Group 1_0
        [3.0, 0.0, 0.0],  # Group 1_1
        [3.5, 0.0, 0.0],  # Group 1_1
        [4.0, 0.0, 0.0],  # Group 1_1
        [4.5, 0.0, 0.0],  # Group 1_1
    ])

    response_mask = torch.ones_like(token_level_rewards)
    index = np.array([0] * 8 + [1] * 8)

    # Compute advantages
    advantages, returns = core_algos.compute_similarity_group_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        similarity_group_labels=similarity_group_labels,
        norm_adv_by_std_in_grpo=True,
    )

    print("\nAdvantages (by group):")
    for i, label in enumerate(similarity_group_labels):
        adv = advantages[i, 0].item()
        reward = token_level_rewards[i].sum().item()
        print(f"  [{label}] Reward: {reward:.2f}, Advantage: {adv:.4f}")

    # Verify within-group normalization
    print("\nVerifying within-group normalization:")
    for group_label in set(similarity_group_labels):
        indices = [i for i, lbl in enumerate(similarity_group_labels) if lbl == group_label]
        group_advs = [advantages[i, 0].item() for i in indices]
        mean_adv = np.mean(group_advs)
        std_adv = np.std(group_advs)
        print(f"  Group {group_label}: mean={mean_adv:.4f}, std={std_adv:.4f}")

    print("\n✓ Advantage computation test passed!")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Similarity Grouper Test Suite")
    print("=" * 80)

    try:
        test_basic_grouping()
        test_edge_cases()
        test_advantage_computation()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
