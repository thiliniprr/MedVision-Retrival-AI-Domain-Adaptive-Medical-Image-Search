# faiss_index_builder.py
"""
Builds and manages the FAISS vector index for efficient
similarity search over medical image embeddings.

NEW:
  - EmbeddingPostProcessor: PCA whitening and dimensionality reduction
  - Integrated into build_index() for better cosine matching
  - Preprocessing params saved/loaded alongside the index
"""
import os
import faiss
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging
import json

from config import PipelineConfig
from clip_finetuner import CLIPFineTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================== #
#  NEW: Embedding Post-Processor
# ================================================================== #

class EmbeddingPostProcessor:
    """
    Applies transformations to embeddings before indexing
    that improve cosine similarity matching.

    Supports:
      - PCA whitening: decorrelates dimensions, equalizes variance
      - Dimensionality reduction: removes noise dimensions
      - L2 normalization

    The processor saves its learned parameters (mean, eigenvectors,
    eigenvalues) so that query embeddings can be transformed identically.
    """

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.whitening_matrix: Optional[np.ndarray] = None
        self.projection_matrix: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(
        self,
        embeddings: np.ndarray,
        whiten: bool = True,
        reduce_dim: Optional[int] = None,
    ):
        """
        Learn transformation parameters from the embedding matrix.

        Args:
            embeddings: (N, D) matrix of embeddings
            whiten: Whether to apply PCA whitening
            reduce_dim: Target dimensionality (None = keep all)
        """
        self.mean = embeddings.mean(axis=0)
        centered = embeddings - self.mean

        if whiten or reduce_dim is not None:
            # Compute covariance and eigen-decomposition
            cov = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Avoid division by zero
            eigenvalues = np.maximum(eigenvalues, 1e-8)

            # ── NEW: Keep only meaningful variance dimensions ──
            if whiten:
                variance_ratio = eigenvalues / eigenvalues.sum()
                cumulative_variance = np.cumsum(variance_ratio)
                n_meaningful = np.searchsorted(cumulative_variance, 0.95) + 1
    
                self.whitening_matrix = (
                    eigenvectors[:, :n_meaningful]
                    @ np.diag(1.0 / np.sqrt(eigenvalues[:n_meaningful]))
                )
                logger.info(
                    f"Whitening: keeping {n_meaningful}/{len(eigenvalues)} "
                    f"dimensions (95% variance)"
                )
            # ── END NEW ──

            if reduce_dim is not None:
                # Keep top-k principal components
                target = min(reduce_dim, embeddings.shape[1])
                if whiten:
                    self.whitening_matrix = self.whitening_matrix[:, :target]
                else:
                    self.projection_matrix = eigenvectors[:, :target]
                logger.info(
                    f"Dimensionality reduction: "
                    f"{embeddings.shape[1]} -> {target}"
                )

        self.is_fitted = True

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply the learned transformation to embeddings.
        Can be used for both index embeddings and query embeddings.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Processor not fitted! Call fit() first."
            )

        centered = embeddings - self.mean

        if self.whitening_matrix is not None:
            transformed = centered @ self.whitening_matrix
        elif self.projection_matrix is not None:
            transformed = centered @ self.projection_matrix
        else:
            transformed = centered

        return transformed.astype(np.float32)

    def fit_transform(
        self,
        embeddings: np.ndarray,
        whiten: bool = True,
        reduce_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(embeddings, whiten=whiten, reduce_dim=reduce_dim)
        return self.transform(embeddings)

    def save(self, save_dir: str):
        """Save preprocessing parameters to disk."""
        params = {
            "mean": self.mean,
            "whitening_matrix": self.whitening_matrix,
            "projection_matrix": self.projection_matrix,
            "is_fitted": self.is_fitted,
        }
        path = os.path.join(save_dir, "preprocessing_params.pkl")
        with open(path, "wb") as f:
            pickle.dump(params, f)
        logger.info(f"Preprocessing params saved: {path}")

    def load(self, save_dir: str):
        """Load preprocessing parameters from disk."""
        path = os.path.join(save_dir, "preprocessing_params.pkl")
        if not os.path.exists(path):
            logger.warning(
                f"No preprocessing params found at {path}. "
                f"Query embeddings will NOT be whitened."
            )
            return

        with open(path, "rb") as f:
            params = pickle.load(f)

        self.mean = params["mean"]
        self.whitening_matrix = params["whitening_matrix"]
        self.projection_matrix = params["projection_matrix"]
        self.is_fitted = params["is_fitted"]
        logger.info(f"Preprocessing params loaded from {path}")


# ================================================================== #
#  FAISS Index Builder
# ================================================================== #

class FAISSIndexBuilder:
    """
    Builds and manages a FAISS index for medical image retrieval.
    Stores embeddings and associated metadata (captions, indices).

    NEW:
      - Integrates EmbeddingPostProcessor for whitening/PCA
      - Saves/loads preprocessing params alongside index
      - transform_query_embedding() for consistent query preprocessing
    """
    
    def verify_index_consistency(self, sample_indices=None, n_samples=100):
        """Verify that indexed embeddings retrieve themselves with score ≈ 1.0"""
        if self.index is None:
            raise RuntimeError("Index not built yet!")

        if sample_indices is None:
            sample_indices = range(min(n_samples, len(self.metadata)))

        low_score_count = 0
        for i in sample_indices:
            emb = self.embeddings[i].reshape(1, -1).copy()
            if self.config.faiss.normalize_embeddings:
                faiss.normalize_L2(emb)
            scores, indices = self.index.search(emb, 1)

            if indices[0][0] != i or scores[0][0] < 0.99:
                low_score_count += 1
                logger.warning(
                    f"Index {i}: self-score={scores[0][0]:.4f}, "
                    f"retrieved_idx={indices[0][0]}"
                )

        logger.info(
            f"Self-retrieval check: {low_score_count}/"
            f"{len(list(sample_indices))} inconsistencies found"
        )
    
    
    def __init__(self, config: PipelineConfig, finetuner: CLIPFineTuner):
        self.config = config
        self.finetuner = finetuner
        self.index = None
        self.metadata: Dict[int, Dict] = {}
        self.embeddings: Optional[np.ndarray] = None

        # NEW: Embedding post-processor
        self.preprocessor = EmbeddingPostProcessor()

    def _create_index(self, embedding_dim: int) -> faiss.Index:
        """Create FAISS index based on configuration."""
        index_type = self.config.faiss.index_type

        if index_type == "Flat":
            if self.config.faiss.normalize_embeddings:
                index = faiss.IndexFlatIP(embedding_dim)
            else:
                index = faiss.IndexFlatL2(embedding_dim)

        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            nlist = min(
                self.config.faiss.nlist,
                max(1, len(self.metadata) // 10),
            )
            index = faiss.IndexIVFFlat(
                quantizer, embedding_dim, nlist,
                faiss.METRIC_INNER_PRODUCT,
            )

        elif index_type == "IVFPQ":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            nlist = min(
                self.config.faiss.nlist,
                max(1, len(self.metadata) // 10),
            )
            m = 8
            nbits = 8
            index = faiss.IndexIVFPQ(
                quantizer, embedding_dim, nlist, m, nbits,
                faiss.METRIC_INNER_PRODUCT,
            )

        elif index_type == "HNSW":
            M = 32
            index = faiss.IndexHNSWFlat(
                embedding_dim, M, faiss.METRIC_INNER_PRODUCT
            )
            index.hnsw.efSearch = 64
            index.hnsw.efConstruction = 128

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        if self.config.faiss.use_gpu and faiss.get_num_gpus() > 0:
            gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
            logger.info("FAISS index moved to GPU")

        return index

    @torch.no_grad()
    def build_index(self, dataset, batch_size: int = 64):
        """
        Build the FAISS index from the dataset by encoding all images.
        NEW: Applies whitening/PCA if configured.
        """
        logger.info("Building FAISS index from dataset...")

        # Collect all embeddings
        all_embeddings = []
        all_captions = []
        all_indices = []

        def collate_fn(batch):
            return {
                "pixel_values": torch.stack(
                    [item["pixel_values"] for item in batch]
                ),
                "input_ids": torch.stack(
                    [item["input_ids"] for item in batch]
                ),
                "attention_mask": torch.stack(
                    [item["attention_mask"] for item in batch]
                ),
                "captions": [item["caption"] for item in batch],
                "indices": [item["index"] for item in batch],
            }

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        self.finetuner.model.eval()

        for batch in tqdm(dataloader, desc="Encoding images for FAISS"):
            pixel_values = batch["pixel_values"].to(self.finetuner.device)
            image_embeds = self.finetuner.encode_image(pixel_values)
            embeddings_np = image_embeds.cpu().numpy()

            all_embeddings.append(embeddings_np)
            all_captions.extend(batch["captions"])
            all_indices.extend(batch["indices"])

        self.embeddings = np.vstack(all_embeddings).astype(np.float32)
        logger.info(f"Raw embeddings shape: {self.embeddings.shape}")

        # ============================================================
        # NEW: Apply embedding post-processing (whitening / PCA)
        # ============================================================
        use_whitening = self.config.faiss.use_whitening
        reduce_dim = (
            self.config.faiss.target_dim
            if self.config.faiss.reduce_dimensions
            else None
        )

        if use_whitening or reduce_dim is not None:
            logger.info(
                f"Post-processing embeddings: "
                f"whiten={use_whitening}, "
                f"reduce_dim={reduce_dim}"
            )
            self.embeddings = self.preprocessor.fit_transform(
                self.embeddings,
                whiten=use_whitening,
                reduce_dim=reduce_dim,
            )
            logger.info(
                f"Post-processed embeddings shape: "
                f"{self.embeddings.shape}"
            )

        # Determine final embedding dimension
        embedding_dim = self.embeddings.shape[1]

        # Normalize
        if self.config.faiss.normalize_embeddings:
            faiss.normalize_L2(self.embeddings)

        # Store metadata
        for i, (caption, idx) in enumerate(
            zip(all_captions, all_indices)
        ):
            self.metadata[i] = {
                "caption": caption,
                "original_index": idx,
            }

        # Create and populate index
        self.index = self._create_index(embedding_dim)

        if hasattr(self.index, "train"):
            if not self.index.is_trained:
                logger.info("Training FAISS index...")
                self.index.train(self.embeddings)

        self.index.add(self.embeddings)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

        if isinstance(self.index, faiss.IndexIVFFlat) or isinstance(
            self.index, faiss.IndexIVFPQ
        ):
            self.index.nprobe = self.config.faiss.nprobe

        return self.index

    def transform_query_embedding(
        self, query_embedding: np.ndarray
    ) -> np.ndarray:
        """
        NEW: Apply the same preprocessing to a query embedding
        that was applied to the index embeddings.

        This ensures consistency between index and query space.
        Must be called on raw query embeddings before search.
        """
        if self.preprocessor.is_fitted:
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = self.preprocessor.transform(query_embedding)
        return query_embedding

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        pre_transformed: bool = False,
    ) -> List[Dict]:
        """
        Search the FAISS index for similar images.

        Args:
            query_embedding: Query vector (1, D) or (D,)
            top_k: Number of results to return
            pre_transformed: If True, skip whitening/PCA transform
                             (embedding is already in index space)
        """
        if self.index is None:
            raise RuntimeError(
                "Index not built yet! Call build_index first."
            )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # ONLY transform if the embedding is still in raw CLIP space
        if not pre_transformed:
            query_embedding = self.transform_query_embedding(
                query_embedding
            )

        # Normalize query
        if self.config.faiss.normalize_embeddings:
            faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            result = {
                "score": float(score),
                "faiss_index": int(idx),
                "caption": self.metadata[idx]["caption"],
                "original_index": self.metadata[idx]["original_index"],
            }
            results.append(result)

        return results

    def save_index(self):
        """Save FAISS index, metadata, and preprocessing params to disk."""
        save_dir = self.config.faiss.index_save_path

        # Save FAISS index
        index_path = os.path.join(save_dir, "faiss.index")
        if self.config.faiss.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)

        # Save metadata
        metadata_path = os.path.join(save_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        # Save embeddings
        embeddings_path = os.path.join(save_dir, "embeddings.npy")
        np.save(embeddings_path, self.embeddings)

        # NEW: Save preprocessing params
        if self.preprocessor.is_fitted:
            self.preprocessor.save(save_dir)

        # Save config info
        config_path = os.path.join(save_dir, "index_config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "index_type": self.config.faiss.index_type,
                    "num_vectors": int(self.index.ntotal),
                    "embedding_dim": int(self.embeddings.shape[1]),
                    "normalize": self.config.faiss.normalize_embeddings,
                    "use_whitening": self.config.faiss.use_whitening,
                    "reduce_dimensions": self.config.faiss.reduce_dimensions,
                    "target_dim": self.config.faiss.target_dim,
                },
                f,
                indent=2,
            )

        logger.info(f"FAISS index saved to {save_dir}")

    def load_index(self):
        """Load FAISS index, metadata, and preprocessing params from disk."""
        save_dir = self.config.faiss.index_save_path

        index_path = os.path.join(save_dir, "faiss.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index found at {index_path}")

        self.index = faiss.read_index(index_path)

        if self.config.faiss.use_gpu and faiss.get_num_gpus() > 0:
            gpu_res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)

        metadata_path = os.path.join(save_dir, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        embeddings_path = os.path.join(save_dir, "embeddings.npy")
        self.embeddings = np.load(embeddings_path)

        # NEW: Load preprocessing params
        self.preprocessor = EmbeddingPostProcessor()
        self.preprocessor.load(save_dir)

        logger.info(
            f"FAISS index loaded: {self.index.ntotal} vectors, "
            f"{len(self.metadata)} metadata entries, "
            f"preprocessor fitted={self.preprocessor.is_fitted}"
        )

    def get_index_stats(self) -> Dict:
        if self.index is None:
            return {"status": "not built"}

        return {
            "total_vectors": int(self.index.ntotal),
            "embedding_dim": (
                int(self.embeddings.shape[1])
                if self.embeddings is not None
                else "unknown"
            ),
            "index_type": self.config.faiss.index_type,
            "metadata_entries": len(self.metadata),
            "preprocessor_fitted": self.preprocessor.is_fitted,
        }