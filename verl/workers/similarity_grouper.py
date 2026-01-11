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
Similarity-based grouping module for creative writing tasks.

This module implements similarity-based response grouping using sentence embeddings
and silhouette coefficient for optimal cluster selection. It's designed to preserve
diversity in creative writing by only normalizing rewards within similar response groups.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import ray
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


@ray.remote
class SimilarityGrouperWorker:
    """
    Ray remote worker for computing similarity-based grouping of responses.

    This worker:
    1. Loads BAAI/bge-m3 model on GPU for efficient embedding computation
    2. Computes embeddings for response texts
    3. Uses silhouette coefficient to determine optimal number of groups (2-7)
    4. Returns group labels for each response

    Designed to run in parallel across multiple GPUs for efficiency.
    Uses Ray's placement group mechanism to share GPUs with actor/RM workers.
    """

    # Minimum silhouette coefficient threshold
    # Below this value, all responses are considered too similar (grouped as 1)
    MIN_SILHOUETTE = 0.1

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialize the similarity grouper worker.

        Args:
            model_name: HuggingFace model identifier for sentence embeddings
        """
        import os

        # Ray sets CUDA_VISIBLE_DEVICES via placement group
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')

        # Determine device
        if torch.cuda.is_available():
            # Ray sets CUDA_VISIBLE_DEVICES, so cuda:0 refers to the assigned GPU
            self.device = 'cuda:0'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[SimilarityGrouperWorker] CUDA_VISIBLE_DEVICES={cuda_visible}")
            print(f"[SimilarityGrouperWorker] Loading model: {model_name} on GPU: {gpu_name}")
        else:
            # Fallback to CPU
            self.device = 'cpu'
            print(f"[SimilarityGrouperWorker] CUDA_VISIBLE_DEVICES={cuda_visible}")
            print(f"[SimilarityGrouperWorker] WARNING: No CUDA available, falling back to CPU")

        self.model = SentenceTransformer(model_name, device=self.device)

        print(f"[SimilarityGrouperWorker] Model loaded successfully on {self.device}")

    def ping(self):
        """Simple health check to verify worker is ready."""
        return "pong"

    def _compute_optimal_grouping(
        self,
        texts: List[str]
    ) -> Tuple[List[int], int, float]:
        """
        Compute optimal grouping for a list of texts using silhouette coefficient.

        Args:
            texts: List of text strings to group (typically 8 responses for one prompt)

        Returns:
            Tuple of (group_labels, n_groups, silhouette_score)
            - group_labels: List of group indices [0, 1, 2, ...] for each text
            - n_groups: Number of groups (1-8)
            - silhouette_score: Quality metric of the grouping
        """
        n = len(texts)

        # Edge case: single text
        if n <= 1:
            return [0], 1, 0.0

        # Encode texts to embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix

        # Try clustering with 2 to min(7, n-1) groups
        # Silhouette coefficient requires at least 2 groups and at most n-1 groups
        max_groups = min(7, n - 1)

        if max_groups < 2:
            # If n < 3, we can only have 1 group
            return [0] * n, 1, 0.0

        best_grouping = None
        best_silhouette = -1.0

        for n_groups in range(2, max_groups + 1):
            # Hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_groups,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)

            # Compute silhouette score
            silhouette = silhouette_score(embeddings, labels, metric='cosine')

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_grouping = labels.tolist()

        # Check if silhouette score is too low (indicating poor clustering quality)
        if best_silhouette < self.MIN_SILHOUETTE:
            # All responses are too similar, treat as 1 group
            return [0] * n, 1, best_silhouette
        else:
            # Use the best grouping found
            n_groups = len(set(best_grouping))
            return best_grouping, n_groups, best_silhouette

    def compute_groups_for_batch(
        self,
        results: List[str],
        uids: List[int],
        k: int = 8
    ) -> Dict:
        """
        Compute similarity groups for a batch of results.

        Args:
            results: List of result texts (already extracted from </think>)
            uids: List of prompt UIDs corresponding to each result
            k: Number of responses per prompt (typically 8)

        Returns:
            Dictionary containing:
                - 'similarity_group_labels': List of group labels (one per result)
                - 'group_stats': Dict with statistics about grouping quality
        """
        # Group results by UID
        uid_to_results = defaultdict(list)
        uid_to_indices = defaultdict(list)

        for idx, (result, uid) in enumerate(zip(results, uids)):
            uid_to_results[uid].append(result)
            uid_to_indices[uid].append(idx)

        # Initialize output
        similarity_group_labels = [0] * len(results)
        group_stats = {
            'n_prompts': len(uid_to_results),
            'silhouette_scores': [],
            'n_groups_per_prompt': [],
        }

        # Process each prompt's responses
        for uid in uid_to_results:
            texts = uid_to_results[uid]
            indices = uid_to_indices[uid]

            # Compute optimal grouping
            group_labels, n_groups, silhouette = self._compute_optimal_grouping(texts)

            # Assign labels (format: "{uid}_{group_id}")
            for local_idx, global_idx in enumerate(indices):
                group_label = f"{uid}_{group_labels[local_idx]}"
                similarity_group_labels[global_idx] = group_label

            # Record statistics
            group_stats['silhouette_scores'].append(silhouette)
            group_stats['n_groups_per_prompt'].append(n_groups)

        return {
            'similarity_group_labels': similarity_group_labels,
            'group_stats': group_stats
        }


def compute_single_member_mask(similarity_group_labels: List[str]) -> np.ndarray:
    """
    Compute a boolean mask indicating which responses belong to single-member groups.

    Single-member groups don't need reward normalization (advantage will be 0 anyway).

    Args:
        similarity_group_labels: List of group labels (e.g., ["0_0", "0_0", "0_1", ...])

    Returns:
        Boolean numpy array of same length, True for single-member groups
    """
    from collections import Counter

    # Count members in each group
    group_counts = Counter(similarity_group_labels)

    # Create mask
    single_member_mask = np.array([
        group_counts[label] == 1
        for label in similarity_group_labels
    ], dtype=bool)

    return single_member_mask


def compute_group_distribution_per_prompt(
    similarity_group_labels: List[str],
    uids: List[int]
) -> Dict[int, int]:
    """
    Compute distribution of number of groups per prompt.

    This function answers: "How many prompts were split into k groups?"

    Args:
        similarity_group_labels: List of group labels (e.g., ["0_0", "0_0", "0_1", ...])
        uids: List of prompt UIDs

    Returns:
        Dictionary mapping n_groups -> count
        Example: {1: 10, 3: 15, 5: 5} means:
            - 10 prompts were split into 1 group
            - 15 prompts were split into 3 groups
            - 5 prompts were split into 5 groups
    """
    from collections import Counter, defaultdict

    # Parse group labels to extract uid and group_id
    uid_to_groups = defaultdict(set)

    for label, uid in zip(similarity_group_labels, uids):
        # label format: "{uid}_{group_id}"
        parts = label.split('_')
        if len(parts) == 2:
            group_id = parts[1]
            uid_to_groups[uid].add(group_id)

    # Count unique groups per prompt
    n_groups_per_prompt = [len(groups) for groups in uid_to_groups.values()]

    # Return distribution
    return dict(Counter(n_groups_per_prompt))
