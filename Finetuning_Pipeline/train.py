# train.py
"""
Standalone CLIP fine-tuning script for MIMIC-CXR.

Features:
  - Automatic 90/10 train/val split
  - Per-epoch training and validation metrics
  - Learning curve plotting
  - Detailed logging with accuracy, loss, retrieval metrics
  - Checkpoint saving (best + periodic)
  - Resume from checkpoint support

Usage:
  python train.py --epochs 10 --batch_size 32 --lr 5e-6
  python train.py --epochs 10 --batch_size 32 --lr 5e-6 --max_samples 1000  # quick test
  python train.py --resume --checkpoint best_model  # resume training
"""

import os
import sys
import argparse
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    CLIPModel,
    CLIPProcessor,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ================================================================== #
#  Logging setup
# ================================================================== #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ================================================================== #
#  Configuration
# ================================================================== #

class TrainConfig:
    """All training hyperparameters in one place."""

    def __init__(self, args):
        # Model
        self.model_name: str = args.model_name
        self.clip_max_tokens: int = 77  # CLIP hard limit

        # Dataset
        self.dataset_name: str = "itsanmolgupta/mimic-cxr-dataset"
        self.image_column: str = "image"
        self.text_columns: List[str] = ["findings"]
        self.val_split_ratio: float = 0.1  # 10% for validation
        self.max_samples: Optional[int] = args.max_samples
        self.num_workers: int = args.num_workers

        # Training
        self.epochs: int = args.epochs
        self.batch_size: int = args.batch_size
        self.learning_rate: float = args.lr
        self.weight_decay: float = args.weight_decay
        self.warmup_ratio: float = args.warmup_ratio
        self.max_grad_norm: float = 1.0
        self.fp16: bool = args.fp16 and torch.cuda.is_available()
        self.gradient_accumulation_steps: int = args.grad_accum

        # Contrastive loss
        self.temperature: float = 0.07
        self.label_smoothing: float = 0.1

        # Freezing strategy
        self.unfreeze_visual_layers: int = args.unfreeze_visual
        self.unfreeze_text_layers: int = args.unfreeze_text

        # Projection head
        self.use_projection_head: bool = args.use_projection
        self.projection_dim: int = args.projection_dim

        # Checkpointing
        self.checkpoint_dir: str = args.checkpoint_dir
        self.save_every_n_epochs: int = args.save_every
        self.output_dir: str = args.output_dir
        self.cache_dir: str = "./cache"

        # Device
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Resume
        self.resume: bool = args.resume
        self.resume_checkpoint: str = args.checkpoint

        # Seed
        self.seed: int = args.seed

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ================================================================== #
#  Dataset
# ================================================================== #

class MIMICCXRClipDataset(Dataset):
    """MIMIC-CXR dataset for CLIP training with findings + impression."""

    CLIP_MAX_TOKENS = 77
    CLIP_USABLE_TOKENS = 75  # minus SOS + EOS

    def __init__(self, hf_dataset, processor: CLIPProcessor):
        self.dataset = hf_dataset
        self.processor = processor
        self.tokenizer = processor.tokenizer

        # Verify columns
        cols = self.dataset.column_names
        logger.info(f"Dataset columns: {cols}, size: {len(self.dataset)}")

    def __len__(self) -> int:
        return len(self.dataset)

    def _clean(self, val) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", "n/a", ""):
            return None
        return " ".join(s.split())

    def _combine_text(self, item: Dict) -> str:
        """Combine findings + impression, fitting within 77 tokens."""
        findings = self._clean(item.get("findings", None))
        impression = self._clean(item.get("impression", None))

        if impression and findings:
            # Try both together
            combined = f"{impression} {findings}"
            tokens = self.tokenizer.encode(combined)
            if len(tokens) <= self.CLIP_USABLE_TOKENS:
                return combined

            # Impression fits, fill remaining with findings
            imp_tokens = self.tokenizer.encode(impression)
            if len(imp_tokens) <= self.CLIP_USABLE_TOKENS:
                remaining = self.CLIP_USABLE_TOKENS - len(imp_tokens) - 1
                if remaining > 5:
                    find_trunc = self._truncate(findings, remaining)
                    return f"{impression} {find_trunc}"
                return impression

            # Even impression too long
            return self._truncate(impression)

        if impression:
            return self._truncate(impression)
        if findings:
            return self._truncate(findings)

        return "Chest radiograph."

    def _truncate(self, text: str, max_tok: Optional[int] = None) -> str:
        max_tok = max_tok or self.CLIP_USABLE_TOKENS
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tok:
            return text
        decoded = self.tokenizer.decode(tokens[:max_tok], skip_special_tokens=True)
        # Try to cut at sentence boundary
        for delim in [". ", "; ", ", "]:
            idx = decoded.rfind(delim)
            if idx > len(decoded) * 0.5:
                decoded = decoded[: idx + 1]
                break
        return decoded.strip()

    def _load_image(self, item: Dict) -> Image.Image:
        raw = item.get("image")
        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, str):
            return Image.open(raw).convert("RGB")
        if isinstance(raw, dict):
            import io as _io
            if "bytes" in raw and raw["bytes"]:
                return Image.open(_io.BytesIO(raw["bytes"])).convert("RGB")
            if "path" in raw and raw["path"]:
                return Image.open(raw["path"]).convert("RGB")
        arr = np.array(raw)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        try:
            image = self._load_image(item)
        except Exception:
            image = Image.new("RGB", (224, 224), "black")

        try:
            caption = self._combine_text(item)
        except Exception:
            caption = "Chest radiograph."

        try:
            proc = self.processor(
                text=caption, images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.CLIP_MAX_TOKENS,
            )
        except Exception:
            proc = self.processor(
                text="Chest radiograph.",
                images=Image.new("RGB", (224, 224), "black"),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.CLIP_MAX_TOKENS,
            )

        return {
            "pixel_values": proc["pixel_values"].squeeze(0),
            "input_ids": proc["input_ids"].squeeze(0),
            "attention_mask": proc["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


# ================================================================== #
#  Model components
# ================================================================== #

class MedicalProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Dropout(0.1),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)


class CLIPContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, label_smoothing=0.1):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))
        self.label_smoothing = label_smoothing

    def forward(self, img_emb, txt_emb):
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        scale = self.logit_scale.exp()
        logits_i2t = scale * img_emb @ txt_emb.t()
        logits_t2i = logits_i2t.t()

        bs = img_emb.size(0)
        labels = torch.arange(bs, device=img_emb.device)

        loss_i2t = F.cross_entropy(logits_i2t, labels, label_smoothing=self.label_smoothing)
        loss_t2i = F.cross_entropy(logits_t2i, labels, label_smoothing=self.label_smoothing)
        loss = (loss_i2t + loss_t2i) / 2.0

        with torch.no_grad():
            acc_i2t = (logits_i2t.argmax(-1) == labels).float().mean()
            acc_t2i = (logits_t2i.argmax(-1) == labels).float().mean()
            # Top-5 accuracy
            _, top5_i2t = logits_i2t.topk(min(5, bs), dim=-1)
            acc_i2t_top5 = (top5_i2t == labels.unsqueeze(1)).any(-1).float().mean()
            _, top5_t2i = logits_t2i.topk(min(5, bs), dim=-1)
            acc_t2i_top5 = (top5_t2i == labels.unsqueeze(1)).any(-1).float().mean()

        return {
            "loss": loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "acc_i2t": acc_i2t,
            "acc_t2i": acc_t2i,
            "acc_i2t_top5": acc_i2t_top5,
            "acc_t2i_top5": acc_t2i_top5,
            "logit_scale": scale,
        }


# ================================================================== #
#  Trainer
# ================================================================== #

class CLIPTrainer:
    """Complete CLIP training pipeline with validation."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Seed
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # ---- Load model & processor ----
        logger.info(f"Loading CLIP: {cfg.model_name}")
        self.model = CLIPModel.from_pretrained(
            cfg.model_name, cache_dir=cfg.cache_dir
        )
        self.processor = CLIPProcessor.from_pretrained(
            cfg.model_name, cache_dir=cfg.cache_dir
        )

        self.clip_proj_dim = self.model.config.projection_dim
        logger.info(f"CLIP projection dim: {self.clip_proj_dim}")

        # ---- Projection heads ----
        self.img_proj = None
        self.txt_proj = None
        if cfg.use_projection_head:
            self.img_proj = MedicalProjectionHead(
                self.clip_proj_dim, cfg.projection_dim
            )
            self.txt_proj = MedicalProjectionHead(
                self.clip_proj_dim, cfg.projection_dim
            )

        # ---- Loss ----
        self.criterion = CLIPContrastiveLoss(
            cfg.temperature, cfg.label_smoothing
        )

        # ---- Freeze ----
        self._setup_freezing()

        # ---- Move to device ----
        self.model.to(self.device)
        if self.img_proj:
            self.img_proj.to(self.device)
            self.txt_proj.to(self.device)
        self.criterion.to(self.device)

        # ---- History ----
        self.history: List[Dict] = []
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.start_epoch = 0

    # ---------------------------------------------------------- #
    #  Freezing
    # ---------------------------------------------------------- #

    def _setup_freezing(self):
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze last N vision layers
        if hasattr(self.model.vision_model, "encoder"):
            layers = self.model.vision_model.encoder.layers
            n = len(layers)
            start = max(0, n - self.cfg.unfreeze_visual_layers)
            for i in range(start, n):
                for p in layers[i].parameters():
                    p.requires_grad = True
            logger.info(f"Vision: unfroze layers {start}–{n-1} of {n}")

        if hasattr(self.model.text_model, "encoder"):
            layers = self.model.text_model.encoder.layers
            n = len(layers)
            start = max(0, n - self.cfg.unfreeze_text_layers)
            for i in range(start, n):
                for p in layers[i].parameters():
                    p.requires_grad = True
            logger.info(f"Text:   unfroze layers {start}–{n-1} of {n}")

        # Projection layers always unfrozen
        for name in ("visual_projection", "text_projection"):
            if hasattr(self.model, name):
                for p in getattr(self.model, name).parameters():
                    p.requires_grad = True

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        extra = 0
        if self.img_proj:
            extra += sum(p.numel() for p in self.img_proj.parameters())
            extra += sum(p.numel() for p in self.txt_proj.parameters())
        extra += sum(p.numel() for p in self.criterion.parameters())

        logger.info(
            f"Parameters — CLIP total: {total:,} | "
            f"CLIP trainable: {trainable:,} ({100*trainable/total:.1f}%) | "
            f"Extra (proj+loss): {extra:,}"
        )

    # ---------------------------------------------------------- #
    #  Feature extraction (robust)
    # ---------------------------------------------------------- #

    def _extract_image_features(self, pixel_values):
        out = self.model.vision_model(pixel_values=pixel_values, return_dict=True)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0, :]
        if hasattr(self.model, "visual_projection") and self.model.visual_projection is not None:
            pooled = self.model.visual_projection(pooled)
        return pooled

    def _extract_text_features(self, input_ids, attention_mask):
        out = self.model.text_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            eos_pos = input_ids.argmax(dim=-1)
            pooled = out.last_hidden_state[
                torch.arange(out.last_hidden_state.size(0), device=input_ids.device),
                eos_pos,
            ]
        if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
            pooled = self.model.text_projection(pooled)
        return pooled

    def _get_embeddings(self, batch):
        pv = batch["pixel_values"].to(self.device)
        ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)

        img_emb = self._extract_image_features(pv)
        txt_emb = self._extract_text_features(ids, mask)

        if self.img_proj:
            img_emb = self.img_proj(img_emb)
            txt_emb = self.txt_proj(txt_emb)

        return img_emb, txt_emb

    # ---------------------------------------------------------- #
    #  Data loading with 90/10 split
    # ---------------------------------------------------------- #

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        logger.info(f"Loading dataset: {self.cfg.dataset_name}")
        raw = load_dataset(
            self.cfg.dataset_name,
            cache_dir=self.cfg.cache_dir,
            trust_remote_code=True,
        )

        # Use first available split
        split_name = list(raw.keys())[0]
        full_data = raw[split_name]
        logger.info(f"Full dataset split '{split_name}': {len(full_data)} samples")
        logger.info(f"Columns: {full_data.column_names}")

        # Subsample if requested
        if self.cfg.max_samples and self.cfg.max_samples < len(full_data):
            full_data = full_data.select(range(self.cfg.max_samples))
            logger.info(f"Subsampled to {len(full_data)} samples")

        # Create PyTorch dataset
        full_dataset = MIMICCXRClipDataset(full_data, self.processor)

        # ----- 90/10 split -----
        total = len(full_dataset)
        val_size = int(total * self.cfg.val_split_ratio)
        train_size = total - val_size

        logger.info(f"Splitting: {train_size} train / {val_size} val "
                     f"({self.cfg.val_split_ratio:.0%} val ratio)")

        generator = torch.Generator().manual_seed(self.cfg.seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches:   {len(val_loader)}")

        return train_loader, val_loader

    # ---------------------------------------------------------- #
    #  Optimizer
    # ---------------------------------------------------------- #

    def _build_optimizer(self, num_steps: int):
        groups = []

        # CLIP backbone (lower LR)
        clip_params = [p for p in self.model.parameters() if p.requires_grad]
        if clip_params:
            groups.append({
                "params": clip_params,
                "lr": self.cfg.learning_rate,
                "weight_decay": self.cfg.weight_decay,
            })

        # Projection heads (higher LR)
        if self.img_proj:
            groups.append({
                "params": self.img_proj.parameters(),
                "lr": self.cfg.learning_rate * 10,
                "weight_decay": self.cfg.weight_decay,
            })
            groups.append({
                "params": self.txt_proj.parameters(),
                "lr": self.cfg.learning_rate * 10,
                "weight_decay": self.cfg.weight_decay,
            })

        # Loss temperature
        groups.append({
            "params": self.criterion.parameters(),
            "lr": self.cfg.learning_rate * 5,
            "weight_decay": 0.0,
        })

        optimizer = AdamW(groups)

        warmup = int(num_steps * self.cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=num_steps,
        )

        logger.info(f"Optimizer: AdamW | Warmup steps: {warmup} | Total steps: {num_steps}")
        return optimizer, scheduler

    # ---------------------------------------------------------- #
    #  Train one epoch
    # ---------------------------------------------------------- #

    def train_one_epoch(
        self, train_loader, optimizer, scheduler, scaler, epoch
    ) -> Dict[str, float]:
        self.model.train()
        if self.img_proj:
            self.img_proj.train()
            self.txt_proj.train()

        losses, i2t_accs, t2i_accs = [], [], []
        i2t_top5s, t2i_top5s = [], []

        pbar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch+1}/{self.cfg.epochs}",
            leave=True,
        )

        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # Forward
            if self.cfg.fp16:
                with autocast():
                    img_emb, txt_emb = self._get_embeddings(batch)
                    out = self.criterion(img_emb, txt_emb)
                    loss = out["loss"] / self.cfg.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                img_emb, txt_emb = self._get_embeddings(batch)
                out = self.criterion(img_emb, txt_emb)
                loss = out["loss"] / self.cfg.gradient_accumulation_steps
                loss.backward()

            # Accumulate & step
            if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                if self.cfg.fp16:
                    scaler.unscale_(optimizer)
                all_params = (
                    [p for p in self.model.parameters() if p.requires_grad]
                    + (list(self.img_proj.parameters()) if self.img_proj else [])
                    + (list(self.txt_proj.parameters()) if self.txt_proj else [])
                )
                torch.nn.utils.clip_grad_norm_(all_params, self.cfg.max_grad_norm)

                if self.cfg.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            # Track
            losses.append(out["loss"].item())
            i2t_accs.append(out["acc_i2t"].item())
            t2i_accs.append(out["acc_t2i"].item())
            i2t_top5s.append(out["acc_i2t_top5"].item())
            t2i_top5s.append(out["acc_t2i_top5"].item())

            pbar.set_postfix({
                "loss": f"{np.mean(losses[-50:]):.4f}",
                "i2t@1": f"{np.mean(i2t_accs[-50:]):.3f}",
                "t2i@1": f"{np.mean(t2i_accs[-50:]):.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        return {
            "train_loss": float(np.mean(losses)),
            "train_acc_i2t": float(np.mean(i2t_accs)),
            "train_acc_t2i": float(np.mean(t2i_accs)),
            "train_acc_i2t_top5": float(np.mean(i2t_top5s)),
            "train_acc_t2i_top5": float(np.mean(t2i_top5s)),
            "train_acc_avg": float(np.mean(i2t_accs + t2i_accs)),
        }

    # ---------------------------------------------------------- #
    #  Validate
    # ---------------------------------------------------------- #

    @torch.no_grad()
    def validate(self, val_loader, epoch) -> Dict[str, float]:
        self.model.eval()
        if self.img_proj:
            self.img_proj.eval()
            self.txt_proj.eval()

        losses, i2t_accs, t2i_accs = [], [], []
        i2t_top5s, t2i_top5s = [], []

        # Also collect all embeddings for global retrieval metrics
        all_img_emb, all_txt_emb = [], []

        pbar = tqdm(
            val_loader,
            desc=f"Val   Epoch {epoch+1}/{self.cfg.epochs}",
            leave=True,
        )

        for batch in pbar:
            img_emb, txt_emb = self._get_embeddings(batch)
            out = self.criterion(img_emb, txt_emb)

            losses.append(out["loss"].item())
            i2t_accs.append(out["acc_i2t"].item())
            t2i_accs.append(out["acc_t2i"].item())
            i2t_top5s.append(out["acc_i2t_top5"].item())
            t2i_top5s.append(out["acc_t2i_top5"].item())

            # Collect for global metrics
            all_img_emb.append(F.normalize(img_emb, dim=-1).cpu())
            all_txt_emb.append(F.normalize(txt_emb, dim=-1).cpu())

            pbar.set_postfix({
                "loss": f"{np.mean(losses):.4f}",
                "i2t@1": f"{np.mean(i2t_accs):.3f}",
                "t2i@1": f"{np.mean(t2i_accs):.3f}",
            })

        # ---- Global retrieval metrics (across entire val set) ----
        all_img_emb = torch.cat(all_img_emb, dim=0)
        all_txt_emb = torch.cat(all_txt_emb, dim=0)
        global_metrics = self._compute_retrieval_metrics(
            all_img_emb, all_txt_emb
        )

        metrics = {
            "val_loss": float(np.mean(losses)),
            "val_acc_i2t": float(np.mean(i2t_accs)),
            "val_acc_t2i": float(np.mean(t2i_accs)),
            "val_acc_i2t_top5": float(np.mean(i2t_top5s)),
            "val_acc_t2i_top5": float(np.mean(t2i_top5s)),
            "val_acc_avg": float(np.mean(i2t_accs + t2i_accs)),
        }
        metrics.update(global_metrics)

        return metrics

    def _compute_retrieval_metrics(
        self, img_emb: torch.Tensor, txt_emb: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute global retrieval metrics:
        R@1, R@5, R@10 for both i2t and t2i over the entire val set.
        """
        n = img_emb.size(0)
        if n == 0:
            return {}

        # Cosine similarity matrix (n × n)
        sim = img_emb @ txt_emb.t()  # already normalized
        labels = torch.arange(n)

        metrics = {}
        for direction, logits in [("i2t", sim), ("t2i", sim.t())]:
            for k in [1, 5, 10]:
                if k > n:
                    continue
                _, topk_idx = logits.topk(k, dim=1)
                hits = (topk_idx == labels.unsqueeze(1)).any(dim=1).float()
                metrics[f"val_recall_{direction}@{k}"] = float(hits.mean())

        # Mean reciprocal rank
        for direction, logits in [("i2t", sim), ("t2i", sim.t())]:
            ranks = (logits.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero(as_tuple=True)[1]
            metrics[f"val_mrr_{direction}"] = float((1.0 / (ranks.float() + 1)).mean())

        return metrics

    # ---------------------------------------------------------- #
    #  Checkpointing
    # ---------------------------------------------------------- #

    def save_checkpoint(self, name: str, epoch: int, optimizer=None, scheduler=None):
        path = os.path.join(self.cfg.checkpoint_dir, name)
        os.makedirs(path, exist_ok=True)

        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

        extra = {
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "config": self.cfg.to_dict(),
        }

        if self.img_proj:
            torch.save(self.img_proj.state_dict(), os.path.join(path, "img_proj.pt"))
            torch.save(self.txt_proj.state_dict(), os.path.join(path, "txt_proj.pt"))
            extra["projection_dim"] = self.cfg.projection_dim
            extra["clip_proj_dim"] = self.clip_proj_dim

        torch.save(self.criterion.state_dict(), os.path.join(path, "criterion.pt"))

        if optimizer:
            extra["optimizer_state"] = optimizer.state_dict()
        if scheduler:
            extra["scheduler_state"] = scheduler.state_dict()

        torch.save(extra, os.path.join(path, "training_state.pt"))
        logger.info(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, name: str):
        path = os.path.join(self.cfg.checkpoint_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.model = CLIPModel.from_pretrained(path)
        self.processor = CLIPProcessor.from_pretrained(path)
        self.model.to(self.device)
        self.clip_proj_dim = self.model.config.projection_dim

        state_path = os.path.join(path, "training_state.pt")
        state = {}
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.start_epoch = state.get("epoch", 0)
            self.best_val_loss = state.get("best_val_loss", float("inf"))
            self.best_val_acc = state.get("best_val_acc", 0.0)
            self.history = state.get("history", [])

        img_proj_path = os.path.join(path, "img_proj.pt")
        if os.path.exists(img_proj_path):
            proj_dim = state.get("projection_dim", self.cfg.projection_dim)
            clip_dim = state.get("clip_proj_dim", self.clip_proj_dim)
            self.img_proj = MedicalProjectionHead(clip_dim, proj_dim).to(self.device)
            self.txt_proj = MedicalProjectionHead(clip_dim, proj_dim).to(self.device)
            self.img_proj.load_state_dict(torch.load(img_proj_path, map_location=self.device))
            self.txt_proj.load_state_dict(
                torch.load(os.path.join(path, "txt_proj.pt"), map_location=self.device)
            )

        logger.info(f"📂 Loaded checkpoint: {path} (epoch {self.start_epoch})")
        return state

    # ---------------------------------------------------------- #
    #  Main training loop
    # ---------------------------------------------------------- #

    def train(self):
        train_loader, val_loader = self.load_data()

        num_steps = (
            len(train_loader) * self.cfg.epochs
            // self.cfg.gradient_accumulation_steps
        )
        optimizer, scheduler = self._build_optimizer(num_steps)

        # Resume
        if self.cfg.resume:
            state = self.load_checkpoint(self.cfg.resume_checkpoint)
            if "optimizer_state" in state:
                optimizer.load_state_dict(state["optimizer_state"])
            if "scheduler_state" in state:
                scheduler.load_state_dict(state["scheduler_state"])

        scaler = GradScaler() if self.cfg.fp16 else None

        # ---- Print header ----
        logger.info("=" * 70)
        logger.info("  CLIP Fine-Tuning on MIMIC-CXR")
        logger.info("=" * 70)
        logger.info(f"  Model:       {self.cfg.model_name}")
        logger.info(f"  Epochs:      {self.cfg.epochs}")
        logger.info(f"  Batch size:  {self.cfg.batch_size}")
        logger.info(f"  LR:          {self.cfg.learning_rate}")
        logger.info(f"  FP16:        {self.cfg.fp16}")
        logger.info(f"  Projection:  {self.cfg.use_projection_head} (dim={self.cfg.projection_dim})")
        logger.info(f"  Device:      {self.device}")
        logger.info(f"  Val ratio:   {self.cfg.val_split_ratio}")
        logger.info("=" * 70)

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()

            # ---- Train ----
            train_metrics = self.train_one_epoch(
                train_loader, optimizer, scheduler, scaler, epoch
            )

            # ---- Validate ----
            val_metrics = self.validate(val_loader, epoch)

            elapsed = time.time() - t0

            # ---- Combine metrics ----
            epoch_metrics = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
                "logit_scale": self.criterion.logit_scale.exp().item(),
                "elapsed_sec": round(elapsed, 1),
            }
            self.history.append(epoch_metrics)

            # ---- Pretty print ----
            logger.info("")
            logger.info("─" * 70)
            logger.info(f"  Epoch {epoch+1}/{self.cfg.epochs}  ({elapsed:.0f}s)")
            logger.info("─" * 70)
            logger.info(
                f"  {'':20s} {'Train':>10s}  {'Val':>10s}"
            )
            logger.info(
                f"  {'Loss':20s} {train_metrics['train_loss']:10.4f}  "
                f"{val_metrics['val_loss']:10.4f}"
            )
            logger.info(
                f"  {'Acc I→T @1':20s} {train_metrics['train_acc_i2t']:10.4f}  "
                f"{val_metrics['val_acc_i2t']:10.4f}"
            )
            logger.info(
                f"  {'Acc T→I @1':20s} {train_metrics['train_acc_t2i']:10.4f}  "
                f"{val_metrics['val_acc_t2i']:10.4f}"
            )
            logger.info(
                f"  {'Acc I→T @5':20s} {train_metrics['train_acc_i2t_top5']:10.4f}  "
                f"{val_metrics['val_acc_i2t_top5']:10.4f}"
            )
            logger.info(
                f"  {'Acc T→I @5':20s} {train_metrics['train_acc_t2i_top5']:10.4f}  "
                f"{val_metrics['val_acc_t2i_top5']:10.4f}"
            )
            logger.info(
                f"  {'Avg Acc @1':20s} {train_metrics['train_acc_avg']:10.4f}  "
                f"{val_metrics['val_acc_avg']:10.4f}"
            )

            # Global retrieval metrics
            for k in ["val_recall_i2t@1", "val_recall_i2t@5", "val_recall_i2t@10",
                       "val_recall_t2i@1", "val_recall_t2i@5", "val_recall_t2i@10",
                       "val_mrr_i2t", "val_mrr_t2i"]:
                if k in val_metrics:
                    label = k.replace("val_", "").replace("_", " ").title()
                    logger.info(f"  {label:20s} {'—':>10s}  {val_metrics[k]:10.4f}")

            logger.info(
                f"  {'Logit scale':20s} {epoch_metrics['logit_scale']:10.4f}"
            )
            logger.info("─" * 70)

            # ---- Save best model ----
            is_best = False
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                is_best = True
            if val_metrics["val_acc_avg"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc_avg"]
                is_best = True

            if is_best:
                self.save_checkpoint("best_model", epoch + 1, optimizer, scheduler)
                logger.info(
                    f"  ★ New best! val_loss={self.best_val_loss:.4f} "
                    f"val_acc={self.best_val_acc:.4f}"
                )

            # ---- Periodic save ----
            if (epoch + 1) % self.cfg.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch+1}", epoch + 1, optimizer, scheduler)

        # ---- Final save ----
        self.save_checkpoint("final_model", self.cfg.epochs, optimizer, scheduler)

        # ---- Save history ----
        self._save_history()

        # ---- Plot ----
        self._plot_curves()

        # ---- Print summary ----
        self._print_summary()

        return self.history

    # ---------------------------------------------------------- #
    #  Reporting
    # ---------------------------------------------------------- #

    def _save_history(self):
        path = os.path.join(self.cfg.output_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"📄 History saved: {path}")

        # Also save as CSV for easy analysis
        csv_path = os.path.join(self.cfg.output_dir, "training_history.csv")
        if self.history:
            import csv
            keys = self.history[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)
            logger.info(f"📄 CSV saved: {csv_path}")

    def _plot_curves(self):
        if not HAS_MATPLOTLIB or not self.history:
            return

        epochs = [h["epoch"] for h in self.history]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("CLIP Fine-Tuning on MIMIC-CXR", fontsize=14, fontweight="bold")

        # ---- 1. Loss ----
        ax = axes[0, 0]
        ax.plot(epochs, [h["train_loss"] for h in self.history], "b-o", label="Train", markersize=4)
        ax.plot(epochs, [h["val_loss"] for h in self.history], "r-o", label="Val", markersize=4)
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Contrastive Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ---- 2. I→T Accuracy @1 ----
        ax = axes[0, 1]
        ax.plot(epochs, [h["train_acc_i2t"] for h in self.history], "b-o", label="Train", markersize=4)
        ax.plot(epochs, [h["val_acc_i2t"] for h in self.history], "r-o", label="Val", markersize=4)
        ax.set_title("Image→Text Accuracy @1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # ---- 3. T→I Accuracy @1 ----
        ax = axes[0, 2]
        ax.plot(epochs, [h["train_acc_t2i"] for h in self.history], "b-o", label="Train", markersize=4)
        ax.plot(epochs, [h["val_acc_t2i"] for h in self.history], "r-o", label="Val", markersize=4)
        ax.set_title("Text→Image Accuracy @1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # ---- 4. Average Accuracy ----
        ax = axes[1, 0]
        ax.plot(epochs, [h["train_acc_avg"] for h in self.history], "b-o", label="Train", markersize=4)
        ax.plot(epochs, [h["val_acc_avg"] for h in self.history], "r-o", label="Val", markersize=4)
        ax.set_title("Average Accuracy @1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # ---- 5. Top-5 Accuracy ----
        ax = axes[1, 1]
        ax.plot(epochs, [h["train_acc_i2t_top5"] for h in self.history], "b--", label="Train I→T @5", markersize=3)
        ax.plot(epochs, [h["val_acc_i2t_top5"] for h in self.history], "r--", label="Val I→T @5", markersize=3)
        ax.plot(epochs, [h["train_acc_t2i_top5"] for h in self.history], "b-.", label="Train T→I @5", markersize=3)
        ax.plot(epochs, [h["val_acc_t2i_top5"] for h in self.history], "r-.", label="Val T→I @5", markersize=3)
        ax.set_title("Top-5 Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # ---- 6. Global Recall ----
        ax = axes[1, 2]
        for k_label, key in [("R@1", "val_recall_i2t@1"), ("R@5", "val_recall_i2t@5"), ("R@10", "val_recall_i2t@10")]:
            vals = [h.get(key, 0) for h in self.history]
            if any(v > 0 for v in vals):
                ax.plot(epochs, vals, "-o", label=f"I→T {k_label}", markersize=4)
        for k_label, key in [("R@1", "val_recall_t2i@1"), ("R@5", "val_recall_t2i@5"), ("R@10", "val_recall_t2i@10")]:
            vals = [h.get(key, 0) for h in self.history]
            if any(v > 0 for v in vals):
                ax.plot(epochs, vals, "--s", label=f"T→I {k_label}", markersize=3)
        ax.set_title("Global Recall (Val)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Recall")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plot_path = os.path.join(self.cfg.output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"📈 Training curves saved: {plot_path}")

    def _print_summary(self):
        if not self.history:
            return

        best_epoch = min(self.history, key=lambda h: h["val_loss"])

        logger.info("")
        logger.info("=" * 70)
        logger.info("  TRAINING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Total epochs:          {len(self.history)}")
        logger.info(f"  Best epoch:            {best_epoch['epoch']}")
        logger.info(f"  Best val loss:         {best_epoch['val_loss']:.4f}")
        logger.info(f"  Best val acc I→T @1:   {best_epoch['val_acc_i2t']:.4f}")
        logger.info(f"  Best val acc T→I @1:   {best_epoch['val_acc_t2i']:.4f}")
        logger.info(f"  Best val avg acc @1:   {best_epoch['val_acc_avg']:.4f}")
        logger.info(f"  Best val acc I→T @5:   {best_epoch['val_acc_i2t_top5']:.4f}")
        logger.info(f"  Best val acc T→I @5:   {best_epoch['val_acc_t2i_top5']:.4f}")

        for key in ["val_recall_i2t@1", "val_recall_i2t@5", "val_recall_i2t@10",
                     "val_mrr_i2t", "val_mrr_t2i"]:
            if key in best_epoch:
                label = key.replace("val_", "").replace("_", " ").title()
                logger.info(f"  Best {label}:   {best_epoch[key]:.4f}")

        logger.info("")

        # First vs last epoch comparison
        first = self.history[0]
        last = self.history[-1]
        logger.info("  Metric                  Epoch 1  →  Final     Change")
        logger.info("  " + "─" * 55)
        for metric in ["val_loss", "val_acc_i2t", "val_acc_t2i", "val_acc_avg"]:
            v1 = first[metric]
            vl = last[metric]
            delta = vl - v1
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            label = metric.replace("val_", "").replace("_", " ").title()
            logger.info(
                f"  {label:24s} {v1:7.4f}  →  {vl:7.4f}    {arrow} {abs(delta):.4f}"
            )

        logger.info("=" * 70)
        logger.info(f"  Checkpoints:  {self.cfg.checkpoint_dir}")
        logger.info(f"  Outputs:      {self.cfg.output_dir}")
        logger.info("=" * 70)


# ================================================================== #
#  CLI
# ================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune CLIP on MIMIC-CXR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32",
                    help="HuggingFace CLIP model name")

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_accum", type=int, default=1,
                    help="Gradient accumulation steps")
    p.add_argument("--fp16", action="store_true", default=True,
                    help="Use mixed precision")
    p.add_argument("--no_fp16", dest="fp16", action="store_false")

    # Data
    p.add_argument("--max_samples", type=int, default=None,
                    help="Max samples (None=all)")
    p.add_argument("--num_workers", type=int, default=4)

    # Model architecture
    p.add_argument("--unfreeze_visual", type=int, default=4,
                    help="Number of visual layers to unfreeze")
    p.add_argument("--unfreeze_text", type=int, default=4,
                    help="Number of text layers to unfreeze")
    p.add_argument("--use_projection", action="store_true", default=True,
                    help="Use medical projection heads")
    p.add_argument("--no_projection", dest="use_projection", action="store_false")
    p.add_argument("--projection_dim", type=int, default=256)

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--save_every", type=int, default=2,
                    help="Save checkpoint every N epochs")

    # Resume
    p.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint")
    p.add_argument("--checkpoint", type=str, default="best_model",
                    help="Checkpoint name for resuming")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(args)

    logger.info(f"Config: {json.dumps(cfg.to_dict(), indent=2, default=str)}")

    trainer = CLIPTrainer(cfg)
    history = trainer.train()

    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()