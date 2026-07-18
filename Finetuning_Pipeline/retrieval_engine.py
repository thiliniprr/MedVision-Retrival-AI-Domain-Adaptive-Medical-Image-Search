# retrieval_engine.py
"""
Handles image query processing, retrieval from FAISS,
and ranking of similar medical images.

NEW:
  - QueryExpander: Rocchio-style pseudo-relevance feedback
  - CrossEncoderReranker: re-scores candidates more precisely
  - MultiModalSearcher: combines image + text queries
  - HardNegativeMiner: mines hard negatives from the index for training
  - RetrievalEngine now orchestrates all enhancement stages
"""
import torch
import numpy as np
import faiss
from PIL import Image
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

from config import PipelineConfig
from clip_finetuner import CLIPFineTuner
from faiss_index_builder import FAISSIndexBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================== #
#  NEW: Query Expander
# ================================================================== #

class QueryExpander:
    """
    Improves retrieval by expanding the query with information
    from initial results (Rocchio-style pseudo-relevance feedback).

    How it works:
      1. Search with the original query
      2. Average the top-k results (assumed relevant)
      3. Create a new query that blends original + feedback centroid
    """

    def __init__(self, index_builder: FAISSIndexBuilder):
        self.index_builder = index_builder

    def expand_query(
        self,
        query_embedding: np.ndarray,
        top_k_initial: int = 10,
        top_k_feedback: int = 3,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> np.ndarray:
        """
        Rocchio-style query expansion.
        Expects query_embedding already in transformed index space.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        results = self.index_builder.search(
            query_embedding,
            top_k=top_k_initial,
            pre_transformed=True,
        )

        if len(results) < top_k_feedback:
            return query_embedding

        feedback_indices = [
            r["faiss_index"] for r in results[:top_k_feedback]
        ]
        feedback_embeddings = self.index_builder.embeddings[
            feedback_indices
        ]

        feedback_centroid = np.mean(
            feedback_embeddings, axis=0, keepdims=True
        )

        expanded = alpha * query_embedding + beta * feedback_centroid
        expanded = expanded.astype(np.float32)

        faiss.normalize_L2(expanded)

        return expanded


# ================================================================== #
#  NEW: Cross-Encoder Re-ranker
# ================================================================== #

class CrossEncoderReranker:
    """
    Re-ranks FAISS results using more precise cosine similarity
    computation.

    Pipeline:
        FAISS (fast, retrieves N candidates)
          -> Re-ranker (precise cosine, picks best K from N)
    """

    def __init__(
        self,
        index_builder: FAISSIndexBuilder,
        finetuner: CLIPFineTuner,
    ):
        self.index_builder = index_builder
        self.finetuner = finetuner

    def rerank(
        self,
        query_embedding: np.ndarray,
        initial_results: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Re-rank initial FAISS results with precise cosine similarity.

        Args:
            query_embedding: The query embedding (1, D) or (D,)
            initial_results: Results from FAISS search
            top_k: How many results to return after re-ranking

        Returns:
            Re-ranked list of results
        """
        if not initial_results:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Normalize query
        query_norm = query_embedding.copy()
        faiss.normalize_L2(query_norm)

        # Get candidate embeddings
        candidate_indices = [r["faiss_index"] for r in initial_results]
        candidate_embeddings = self.index_builder.embeddings[
            candidate_indices
        ].copy()

        # Normalize candidates
        faiss.normalize_L2(candidate_embeddings)

        # Compute precise cosine similarities (double precision for accuracy)
        similarities = np.dot(
            candidate_embeddings.astype(np.float64),
            query_norm.astype(np.float64).T,
        ).squeeze()

        # Handle single-result case
        if similarities.ndim == 0:
            similarities = np.array([float(similarities)])

        # Sort by refined scores (descending)
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        reranked_results = []
        for new_rank, rank_idx in enumerate(ranked_indices):
            candidate = initial_results[rank_idx].copy()
            candidate["score"] = float(similarities[rank_idx])
            candidate["rank"] = new_rank + 1
            candidate["reranked"] = True
            reranked_results.append(candidate)

        logger.info(
            f"Re-ranked {len(initial_results)} candidates -> "
            f"top {len(reranked_results)} results"
        )

        return reranked_results


# ================================================================== #
#  NEW: Multi-Modal Searcher
# ================================================================== #

class MultiModalSearcher:
    """
    Combines image and text embeddings for more accurate search.
    A radiologist can provide both an image AND a text description
    like "bilateral pleural effusion with cardiomegaly".

    Because CLIP aligns image and text in the same embedding space,
    we can create a weighted combination of both modalities.
    """

    def __init__(
        self,
        index_builder: FAISSIndexBuilder,
        finetuner: CLIPFineTuner,
    ):
        self.index_builder = index_builder
        self.finetuner = finetuner

    @torch.no_grad()
    def create_multimodal_query(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        image_weight: float = 0.6,
        text_weight: float = 0.4,
    ) -> np.ndarray:
        """
        Create a combined query embedding from image and/or text.

        Args:
            image: Optional PIL Image
            text: Optional text string
            image_weight: Weight for image embedding
            text_weight: Weight for text embedding

        Returns:
            Combined, normalized query embedding (1, D)
        """
        self.finetuner.model.eval()
        embeddings = []
        weights = []

        if image is not None:
            image = image.convert("RGB")
            processed = self.finetuner.processor(
                images=image, return_tensors="pt"
            )
            pixel_values = processed["pixel_values"].to(
                self.finetuner.device
            )
            image_emb = self.finetuner.encode_image(pixel_values)
            embeddings.append(image_emb.cpu().numpy())
            weights.append(image_weight)

        if text is not None and text.strip():
            text_emb = self.finetuner.encode_text_from_string(text)
            embeddings.append(text_emb.cpu().numpy())
            weights.append(text_weight)

        if not embeddings:
            raise ValueError("Provide at least one of image or text")

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Weighted combination
        combined = np.zeros_like(embeddings[0])
        for emb, w in zip(embeddings, weights):
            emb_copy = emb.copy()
            faiss.normalize_L2(emb_copy)
            combined += w * emb_copy

        combined = combined.astype(np.float32)

        # Normalize the combined query
        faiss.normalize_L2(combined)

        return combined


# ================================================================== #
#  NEW: Hard Negative Miner
# ================================================================== #

class HardNegativeMiner:
    """
    Mines hard negatives from the FAISS index for improved training.

    Hard negatives are images that are close in embedding space but
    semantically different. These are the most informative training
    examples because they force the model to learn fine-grained
    distinctions.
    """

    def __init__(self, index_builder: FAISSIndexBuilder):
        self.index_builder = index_builder

    def mine_hard_negatives(
        self,
        query_embedding: np.ndarray,
        positive_index: int,
        num_negatives: int = 5,
        search_k: int = 50,
    ) -> List[int]:
        """
        Find hard negatives: high similarity but wrong match.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        results = self.index_builder.search(
            query_embedding,
            top_k=search_k,
            pre_transformed=True,
        )

        hard_negatives = []
        for result in results:
            if result["original_index"] == positive_index:
                continue
            hard_negatives.append(result["original_index"])
            if len(hard_negatives) >= num_negatives:
                break

        return hard_negatives

    def mine_for_dataset(
        self,
        num_negatives: int = 5,
        search_k: int = 50,
        max_samples: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        """
        Mine hard negatives for every sample in the index.
        Searches FAISS directly since stored embeddings are
        already in transformed/index space.
        """
        if self.index_builder.embeddings is None:
            raise RuntimeError(
                "Index not built — no embeddings to mine from"
            )

        n_total = len(self.index_builder.metadata)
        if max_samples is not None:
            n_total = min(n_total, max_samples)

        hard_neg_map = {}

        for i in tqdm(range(n_total), desc="Mining hard negatives"):
            emb = self.index_builder.embeddings[i].reshape(1, -1).copy()
            emb = emb.astype(np.float32)

            # Normalize (same as what search() would do)
            if self.index_builder.config.faiss.normalize_embeddings:
                faiss.normalize_L2(emb)

            # Search FAISS directly — embeddings are already
            # in index space, no transform needed
            scores, indices = self.index_builder.index.search(
                emb, search_k
            )

            original_idx = (
                self.index_builder.metadata[i]["original_index"]
            )

            negatives = []
            for idx in indices[0]:
                if idx == -1:
                    continue
                candidate_idx = self.index_builder.metadata[
                    int(idx)
                ]["original_index"]
                # Skip self-match
                if candidate_idx == original_idx:
                    continue
                negatives.append(candidate_idx)
                if len(negatives) >= num_negatives:
                    break

            hard_neg_map[original_idx] = negatives

        logger.info(
            f"Mined hard negatives for {len(hard_neg_map)} samples "
            f"({num_negatives} negatives each)"
        )

        return hard_neg_map


# ================================================================== #
#  Main Retrieval Engine — ENHANCED
# ================================================================== #

class RetrievalEngine:
    """
    Processes query images and retrieves similar cases from FAISS.

    NEW: Orchestrates the full enhanced retrieval pipeline:
      1. Encode query (image and/or text)
      2. Query expansion (optional)
      3. FAISS search (over-retrieve if re-ranking)
      4. Re-ranking (optional)
    """

    def __init__(
        self,
        config: PipelineConfig,
        finetuner: CLIPFineTuner,
        index_builder: FAISSIndexBuilder,
    ):
        self.config = config
        self.finetuner = finetuner
        self.index_builder = index_builder

        # NEW: Initialize search enhancers
        self.query_expander = QueryExpander(index_builder)
        self.reranker = CrossEncoderReranker(index_builder, finetuner)
        self.multimodal_searcher = MultiModalSearcher(
            index_builder, finetuner
        )
        self.hard_negative_miner = HardNegativeMiner(index_builder)

    @torch.no_grad()
    def encode_query_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a query image into an embedding using the fine-tuned CLIP.

        Args:
            image: PIL Image to encode

        Returns:
            Normalized embedding as numpy array (1, D)
        """
        
        self.finetuner.model.eval()
        image = image.convert("RGB")
        processed = self.finetuner.processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = processed["pixel_values"].to(self.finetuner.device)
        embedding = self.finetuner.encode_image(pixel_values)

        return embedding.cpu().numpy()

    def retrieve_similar(
        self,
        query_image: Image.Image,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        text_query: Optional[str] = None,
        use_query_expansion: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        use_multimodal: Optional[bool] = None,
    ) -> List[Dict]:
        if top_k is None:
            top_k = self.config.retrieval.top_k
        if threshold is None:
            threshold = self.config.retrieval.similarity_threshold

        rc = self.config.retrieval
        _use_expansion = (
            use_query_expansion
            if use_query_expansion is not None
            else rc.use_query_expansion
        )
        _use_reranking = (
            use_reranking
            if use_reranking is not None
            else rc.use_reranking
        )
        _use_multimodal = (
            use_multimodal
            if use_multimodal is not None
            else rc.use_multimodal_search
        )

        # Step 1: Build RAW query embedding
        if (
            _use_multimodal
            and text_query is not None
            and text_query.strip()
        ):
            logger.info(
                f"Multi-modal query: image + "
                f"text='{text_query[:50]}...'"
            )
            query_embedding = (
                self.multimodal_searcher.create_multimodal_query(
                    image=query_image,
                    text=text_query,
                    image_weight=rc.image_weight,
                    text_weight=rc.text_weight,
                )
            )
        else:
            query_embedding = self.encode_query_image(query_image)

        # Step 1b: Transform to index space ONCE, EARLY
        query_embedding = (
            self.index_builder.transform_query_embedding(
                query_embedding
            )
        )

        # Step 2: Query expansion (already in transformed space)
        if _use_expansion:
            logger.info(
                f"Applying query expansion "
                f"(rounds={rc.expansion_rounds}, "
                f"feedback_k={rc.expansion_top_k_feedback})"
            )
            for _ in range(rc.expansion_rounds):
                query_embedding = self.query_expander.expand_query(
                    query_embedding,
                    top_k_feedback=rc.expansion_top_k_feedback,
                    alpha=rc.expansion_alpha,
                    beta=rc.expansion_beta,
                )

        # Step 3: FAISS search (already transformed)
        search_k = (
            rc.rerank_candidate_pool if _use_reranking else top_k
        )
        results = self.index_builder.search(
            query_embedding,
            top_k=search_k,
            pre_transformed=True,
        )

        # Step 4: Re-ranking
        if _use_reranking and len(results) > top_k:
            logger.info(
                f"Re-ranking {len(results)} candidates "
                f"-> top {top_k}"
            )
            results = self.reranker.rerank(
                query_embedding,
                initial_results=results,
                top_k=top_k,
            )
        else:
            results = results[:top_k]
            for i, r in enumerate(results):
                r["rank"] = i + 1

        # Step 5: Filter by threshold
        filtered_results = [
            r for r in results if r["score"] >= threshold
        ]

        if not filtered_results:
            logger.warning(
                f"No results above threshold {threshold}. "
                f"Returning top result regardless."
            )
            filtered_results = (
                results[:1] if results else []
            )

        for i, result in enumerate(filtered_results):
            if "rank" not in result:
                result["rank"] = i + 1

        if filtered_results:
            logger.info(
                f"Retrieved {len(filtered_results)} similar "
                f"cases (top score: "
                f"{filtered_results[0]['score']:.4f}, "
                f"expansion={_use_expansion}, "
                f"reranking={_use_reranking}, "
                f"multimodal={_use_multimodal})"
            )

        return filtered_results
        
        
    def batch_retrieve(
        self,
        images: List[Image.Image],
        top_k: Optional[int] = None,
    ) -> List[List[Dict]]:
        """Retrieve similar cases for multiple query images."""
        all_results = []
        for img in images:
            results = self.retrieve_similar(img, top_k=top_k)
            all_results.append(results)
        return all_results

    def mine_hard_negatives_for_training(
        self,
        num_negatives: Optional[int] = None,
        search_k: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        """
        NEW: Convenience method to mine hard negatives for the
        entire indexed dataset. Used between training epochs.

        Args:
            num_negatives: Overrides config.finetune.num_hard_negatives
            search_k: Overrides config.finetune.hard_negative_search_k
            max_samples: Limit to first N samples

        Returns:
            Dict mapping original_index -> [hard negative original_indices]
        """
        if num_negatives is None:
            num_negatives = self.config.finetune.num_hard_negatives
        if search_k is None:
            search_k = self.config.finetune.hard_negative_search_k

        return self.hard_negative_miner.mine_for_dataset(
            num_negatives=num_negatives,
            search_k=search_k,
            max_samples=max_samples,
        )