# dataset_loader.py
"""
Loads and preprocesses the MIMIC-CXR dataset from HuggingFace.
Combines 'findings' and 'impression' columns into a single caption.
Handles CLIP's 77-token limit with smart truncation.

NEW: Supports hard negative sampling for improved contrastive training.
"""
import io
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPTokenizer
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

from config import PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIMICCXRDataset(Dataset):
    """
    Custom Dataset for MIMIC-CXR image + findings/impression pairs.
    Combines multiple text columns into one caption for CLIP.
    Handles CLIP's 77-token limit with smart truncation strategies.

    NEW: Supports hard negative sampling when enabled via
         enable_hard_negative_sampling().
    """

    # CLIP's hard token limit (includes [SOS] and [EOS] tokens)
    CLIP_MAX_TOKENS = 77
    # Usable tokens = 77 - 2 (SOS + EOS) = 75
    CLIP_USABLE_TOKENS = 75

    def __init__(
        self,
        hf_dataset,
        processor: CLIPProcessor,
        config: PipelineConfig,
        split: str = "train",
        augment: bool = False,
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.config = config
        self.split = split
        self.augment = augment 

        # Get the tokenizer for pre-checking token counts
        self.tokenizer = processor.tokenizer

        # NEW: Hard negative sampling state
        self._hard_negative_map: Optional[Dict[int, List[int]]] = None
        self._use_hard_negatives: bool = False

        available_columns = self.dataset.column_names
        logger.info(f"[{split}] Dataset columns: {available_columns}")
        logger.info(f"[{split}] Dataset size: {len(self.dataset)}")
        logger.info(f"[{split}] Augmentations: {self.augment}")

        # ---------- Resolve image column ----------
        self.image_column = self._resolve_image_column(available_columns)

        # ---------- Resolve text columns ----------
        self.text_columns = self._resolve_text_columns(available_columns)

        logger.info(f"[{split}] image_column = '{self.image_column}'")
        logger.info(f"[{split}] text_columns = {self.text_columns}")
        logger.info(
            f"[{split}] CLIP max tokens = {self.CLIP_MAX_TOKENS} "
            f"(usable: {self.CLIP_USABLE_TOKENS})"
        )

        # ---------- Log samples ----------
        self._log_samples()

    # ================================================================ #
    #  NEW: Hard negative sampling support
    # ================================================================ #

    def enable_hard_negative_sampling(
        self, hard_negative_map: Dict[int, List[int]]
    ):
        """
        Enable hard negative sampling for contrastive training.

        Args:
            hard_negative_map: Dict mapping anchor_index → [list of
                               hard negative dataset indices].
                               Built by HardNegativeMiner after an
                               initial FAISS index is created.
        """
        self._hard_negative_map = hard_negative_map
        self._use_hard_negatives = True
        logger.info(
            f"Hard negative sampling enabled: "
            f"{len(hard_negative_map)} anchors with negatives"
        )

    def disable_hard_negative_sampling(self):
        """Disable hard negative sampling."""
        self._use_hard_negatives = False
        logger.info("Hard negative sampling disabled")

    @property
    def has_hard_negatives(self) -> bool:
        return (
            self._use_hard_negatives
            and self._hard_negative_map is not None
            and len(self._hard_negative_map) > 0
        )

    def _get_hard_negative(self, idx: int) -> Optional[Dict]:
        """
        Get a random hard negative sample for the given anchor index.
        Returns None if no hard negatives are available for this index.
        """
        if not self.has_hard_negatives:
            return None

        if idx not in self._hard_negative_map:
            return None

        neg_indices = self._hard_negative_map[idx]
        if not neg_indices:
            return None

        # Pick a random hard negative
        neg_idx = random.choice(neg_indices)

        # Clamp to valid range
        neg_idx = min(neg_idx, len(self.dataset) - 1)

        return self._get_base_item(neg_idx)

    def _get_base_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single item (image + text) without hard negative logic.
        Factored out so hard negatives can reuse this.
        """
        item = self.dataset[idx]

        # ---------- Load image ----------
        try:
            image = self._load_image(item)
        except Exception as e:
            logger.warning(f"Image error at idx {idx}: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        # ---------- Combine text columns (smart truncation) ----------
        try:
            caption = self._combine_text_smart(item)
        except Exception as e:
            logger.warning(f"Text error at idx {idx}: {e}")
            caption = "Chest radiograph."

        # ---------- CLIP processor ----------
        try:
            processed = self.processor(
                text=caption,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.CLIP_MAX_TOKENS,
            )
        except Exception as e:
            logger.warning(f"Processor error at idx {idx}: {e}")
            processed = self.processor(
                text="Chest radiograph.",
                images=Image.new("RGB", (224, 224), color="black"),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.CLIP_MAX_TOKENS,
            )

        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
            "caption": caption,
            "index": idx,
        }

    # ================================================================ #
    #  Column resolution (unchanged)
    # ================================================================ #

    def _resolve_image_column(self, columns: List[str]) -> str:
        if self.config.dataset.image_column in columns:
            return self.config.dataset.image_column
        candidates = ["image", "img", "Image", "xray", "chest_xray"]
        for c in candidates:
            if c in columns:
                return c
        raise KeyError(f"Cannot find image column. Available: {columns}")

    def _resolve_text_columns(self, columns: List[str]) -> List[str]:
        resolved = []
        for col in self.config.dataset.text_columns:
            if col in columns:
                resolved.append(col)
            else:
                for actual_col in columns:
                    if col.lower() == actual_col.lower():
                        resolved.append(actual_col)
                        break
        if not resolved:
            fallbacks = ["findings", "impression", "report", "caption", "text"]
            for fb in fallbacks:
                if fb in columns:
                    resolved.append(fb)
                    break
        if not resolved:
            raise KeyError(
                f"Cannot find any text column.\n"
                f"  Configured: {self.config.dataset.text_columns}\n"
                f"  Available:  {columns}"
            )
        return resolved

    def _log_samples(self):
        """Print first few samples with token counts for debugging."""
        for i in range(min(2, len(self.dataset))):
            try:
                sample = self.dataset[i]
                logger.info(f"[{self.split}] --- Sample {i} ---")
                for col in self.text_columns:
                    val = sample.get(col, None)
                    if val:
                        tokens = self.tokenizer.encode(str(val))
                        logger.info(
                            f"  {col} ({len(tokens)} tokens): "
                            f"{str(val)[:120]}..."
                        )
                    else:
                        logger.info(f"  {col}: <empty>")

                combined = self._combine_text_smart(sample)
                combined_tokens = self.tokenizer.encode(combined)
                logger.info(
                    f"  COMBINED ({len(combined_tokens)} tokens): "
                    f"{combined[:200]}..."
                )
            except Exception as e:
                logger.warning(f"Could not log sample {i}: {e}")

    # ---------------------------------------------------------------- #
    #  Smart text combination for CLIP's 77-token limit (unchanged)
    # ---------------------------------------------------------------- #

    def _combine_text_smart(self, item: Dict) -> str:
        findings = self._clean_text(item.get("findings", None))
        impression = self._clean_text(item.get("impression", None))

        other_texts = {}
        for col in self.text_columns:
            if col not in ("findings", "impression"):
                other_texts[col] = self._clean_text(item.get(col, None))

        if impression and findings:
            return self._fit_to_token_limit(
                impression=impression,
                findings=findings,
            )

        if impression:
            return self._truncate_to_tokens(impression)

        if findings:
            return self._truncate_to_tokens(findings)

        for col, text in other_texts.items():
            if text:
                return self._truncate_to_tokens(text)

        return "Chest radiograph."

    def _fit_to_token_limit(self, impression: str, findings: str) -> str:
        combined = f"{impression} {findings}"
        combined_tokens = self.tokenizer.encode(combined)
        if len(combined_tokens) <= self.CLIP_USABLE_TOKENS:
            return combined

        imp_tokens = self.tokenizer.encode(impression)

        if len(imp_tokens) <= self.CLIP_USABLE_TOKENS:
            remaining = self.CLIP_USABLE_TOKENS - len(imp_tokens) - 1
            if remaining > 5:
                truncated_findings = self._truncate_to_tokens(
                    findings, max_tokens=remaining
                )
                return f"{impression} {truncated_findings}"
            else:
                return impression
        else:
            return self._truncate_to_tokens(impression)

    def _truncate_to_tokens(
        self, text: str, max_tokens: Optional[int] = None
    ) -> str:
        if max_tokens is None:
            max_tokens = self.CLIP_USABLE_TOKENS

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(
            truncated_tokens, skip_special_tokens=True
        )

        for delimiter in [". ", "; ", ", "]:
            last_idx = truncated_text.rfind(delimiter)
            if last_idx > len(truncated_text) * 0.5:
                truncated_text = truncated_text[: last_idx + 1]
                break

        return truncated_text.strip()

    def _clean_text(self, value) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in ("nan", "none", "n/a", ""):
            return None
        text = " ".join(text.split())
        return text

    # ---------------------------------------------------------------- #
    #  Image loading (unchanged)
    # ---------------------------------------------------------------- #

    def _load_image(self, item: Dict) -> Image.Image:
        raw = item[self.image_column]

        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, str):
            return Image.open(raw).convert("RGB")
        if isinstance(raw, dict):
            if "bytes" in raw and raw["bytes"]:
                return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
            if "path" in raw and raw["path"]:
                return Image.open(raw["path"]).convert("RGB")
        try:
            arr = np.array(raw)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        except Exception:
            pass
        raise ValueError(f"Cannot convert image of type {type(raw)}")

    # ---------------------------------------------------------------- #
    #  __getitem__  — MODIFIED to support hard negatives
    # ---------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with the standard fields plus optionally a
        'hard_negative' sub-dict when hard negative sampling is enabled.
        """
        item = self._get_base_item(idx)

        # NEW: Attach hard negative if available
        if self.has_hard_negatives:
            hard_neg = self._get_hard_negative(idx)
            if hard_neg is not None:
                item["hard_negative"] = hard_neg

        return item


# ==================================================================== #
#  DataLoader Factory — MODIFIED for hard negative collation
# ==================================================================== #

class DataLoaderFactory:
    """Creates train/val DataLoaders from the HuggingFace dataset."""

    def __init__(self, config: PipelineConfig, processor: CLIPProcessor):
        self.config = config
        self.processor = processor
        self._raw_dataset = None

    def _load_raw(self):
        if self._raw_dataset is not None:
            return self._raw_dataset

        logger.info(f"Loading dataset: {self.config.dataset.dataset_name}")
        self._raw_dataset = load_dataset(
            self.config.dataset.dataset_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )
        logger.info(f"Splits: {list(self._raw_dataset.keys())}")
        first = list(self._raw_dataset.keys())[0]
        logger.info(f"Columns: {self._raw_dataset[first].column_names}")
        logger.info(f"Size: {len(self._raw_dataset[first])}")
        return self._raw_dataset

    def _resolve_split(self, desired: str) -> Optional[str]:
        raw = self._load_raw()
        if desired in raw:
            return desired
        for name in raw.keys():
            if desired.lower() in name.lower():
                return name
        return None

    def load_datasets(
        self,
    ) -> Tuple[MIMICCXRDataset, Optional[MIMICCXRDataset]]:
        raw = self._load_raw()

        train_name = self._resolve_split(self.config.dataset.train_split)
        if train_name is None:
            train_name = list(raw.keys())[0]

        train_data = raw[train_name]
        if self.config.dataset.max_train_samples:
            n = min(self.config.dataset.max_train_samples, len(train_data))
            train_data = train_data.select(range(n))

        train_dataset = MIMICCXRDataset(
            train_data, self.processor, self.config, split="train", augment=True,
        )

        val_dataset = None
        val_name = self._resolve_split(self.config.dataset.val_split)
        if val_name is not None and val_name != train_name:
            val_data = raw[val_name]
            if self.config.dataset.max_val_samples:
                n = min(self.config.dataset.max_val_samples, len(val_data))
                val_data = val_data.select(range(n))
            val_dataset = MIMICCXRDataset(
                val_data, self.processor, self.config, split="val", augment=False, 
            )
        else:
            logger.warning("No separate val split found.")

        return train_dataset, val_dataset

    @staticmethod
    def collate_fn(batch):
        """
        Standard collation. Also collates hard negatives if present.
        """
        result = {
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

        # NEW: Collate hard negatives if any items have them
        has_hn = [item for item in batch if "hard_negative" in item]
        if has_hn and len(has_hn) == len(batch):
            # All items have hard negatives — batch them
            result["hard_negative"] = {
                "pixel_values": torch.stack(
                    [item["hard_negative"]["pixel_values"] for item in batch]
                ),
                "input_ids": torch.stack(
                    [item["hard_negative"]["input_ids"] for item in batch]
                ),
                "attention_mask": torch.stack(
                    [item["hard_negative"]["attention_mask"] for item in batch]
                ),
            }
        elif has_hn:
            # Partial — only some items have hard negatives.
            # For simplicity, skip hard negatives for this batch.
            pass

        return result

    def get_dataloaders(
        self,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        train_dataset, val_dataset = self.load_datasets()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.finetune.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.finetune.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=self.collate_fn,
            )

        return train_loader, val_loader

    def get_full_dataset_for_indexing(self) -> MIMICCXRDataset:
        raw = self._load_raw()
        train_name = self._resolve_split(self.config.dataset.train_split)
        if train_name is None:
            train_name = list(raw.keys())[0]
        return MIMICCXRDataset(
            raw[train_name], self.processor, self.config, split="index",augment=False,
        )