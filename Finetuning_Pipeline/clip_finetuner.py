# clip_finetuner.py
"""
Fine-tunes CLIP on MIMIC-CXR medical image-text pairs.
Implements contrastive learning with medical domain adaptation.

NEW:
  - ImprovedProjectionHead: multi-layer MLP with BatchNorm
  - HardNegativeContrastiveLoss: margin-based hard negative loss
  - Combined training loop supporting hard negatives
  - encode_text_from_string() for multi-modal search
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    CLIPModel,
    CLIPProcessor,
    get_cosine_schedule_with_warmup,
)
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List
import logging
import json

from config import PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================== #
#  Projection Heads
# ================================================================== #

class MedicalProjectionHead(nn.Module):
    """Original simple projection head (kept for backward compatibility)."""

    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Dropout(0.1),
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ImprovedProjectionHead(nn.Module):
    """
    NEW: Multi-layer projection head with BatchNorm and deeper capacity.
    Projects CLIP embeddings into a space better suited for cosine
    similarity matching in the medical domain.

    Key improvements over MedicalProjectionHead:
      - Multiple hidden layers for more expressive mapping
      - BatchNorm for training stability
      - GELU activation throughout
      - Configurable depth and width
      - L2-normalized output (dot product = cosine similarity)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        # Final layer — no activation, will be L2-normalized downstream
        layers.extend([
            nn.Linear(current_dim, output_dim),
            nn.LayerNorm(output_dim),
        ])

        self.projection = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


# ================================================================== #
#  Loss Functions
# ================================================================== #

class CLIPContrastiveLoss(nn.Module):
    """Symmetric contrastive loss for CLIP fine-tuning."""

    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.1):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(np.log(1.0 / temperature))
        )
        self.label_smoothing = label_smoothing

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        logit_scale = self.temperature.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device)

        loss_i2t = F.cross_entropy(
            logits_per_image, labels, label_smoothing=self.label_smoothing
        )
        loss_t2i = F.cross_entropy(
            logits_per_text, labels, label_smoothing=self.label_smoothing
        )

        total_loss = (loss_i2t + loss_t2i) / 2.0

        with torch.no_grad():
            i2t_acc = (logits_per_image.argmax(dim=-1) == labels).float().mean()
            t2i_acc = (logits_per_text.argmax(dim=-1) == labels).float().mean()

        return {
            "loss": total_loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "i2t_accuracy": i2t_acc,
            "t2i_accuracy": t2i_acc,
            "logit_scale": logit_scale,
        }


class HardNegativeContrastiveLoss(nn.Module):
    """
    NEW: Contrastive loss that emphasizes hard negatives.
    Forces the model to distinguish between genuinely similar
    but semantically different medical images.

    Combines InfoNCE with a margin-based penalty on hard negatives
    that are too close to the anchor.
    """

    def __init__(self, margin: float = 0.2, temperature: float = 0.05):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        hard_negatives: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        hard_negatives = F.normalize(hard_negatives, dim=-1)

        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        neg_sim = torch.sum(anchor * hard_negatives, dim=-1) / self.temperature
        in_batch_sim = (anchor @ positive.t()) / self.temperature

        logits = torch.cat([
            pos_sim.unsqueeze(-1),
            neg_sim.unsqueeze(-1),
            in_batch_sim,
        ], dim=-1)

        labels = torch.zeros(
            logits.size(0), dtype=torch.long, device=logits.device
        )

        infonce_loss = F.cross_entropy(logits, labels)

        margin_violations = F.relu(
            neg_sim - pos_sim + self.margin * (1.0 / self.temperature)
        )
        margin_loss = margin_violations.mean()

        total_loss = infonce_loss + 0.5 * margin_loss

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            avg_pos_sim = (pos_sim * self.temperature).mean()
            avg_neg_sim = (neg_sim * self.temperature).mean()

        return {
            "loss": total_loss,
            "infonce_loss": infonce_loss,
            "margin_loss": margin_loss,
            "accuracy": acc,
            "avg_pos_sim": avg_pos_sim,
            "avg_neg_sim": avg_neg_sim,
        }


# ================================================================== #
#  Main Fine-Tuner
# ================================================================== #

class CLIPFineTuner:
    """
    Handles CLIP fine-tuning pipeline for medical images.

    NEW features:
      - Improved multi-layer projection head option
      - Hard negative contrastive loss
      - Combined training loop (standard + hard negative)
      - encode_text_from_string() for text-based queries
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading CLIP model: {config.model.model_name}")
        self.model = CLIPModel.from_pretrained(
            config.model.model_name,
            cache_dir=config.cache_dir,
        )
        self.processor = CLIPProcessor.from_pretrained(
            config.model.model_name,
            cache_dir=config.cache_dir,
        )

        self.clip_embedding_dim = self.model.config.projection_dim
        logger.info(f"CLIP projection dim: {self.clip_embedding_dim}")

        # Build projection heads
        self.image_projection = None
        self.text_projection = None
        if config.finetune.use_projection_head:
            self.image_projection, self.text_projection = (
                self._build_projection_heads()
            )

        # Loss functions
        self.criterion = CLIPContrastiveLoss(
            temperature=config.finetune.temperature
        )
        self.hard_negative_criterion = HardNegativeContrastiveLoss(
            margin=config.finetune.margin,
            temperature=config.finetune.loss_temperature,
        )

        self._setup_freezing()

        # Move to device
        self.model.to(self.device)
        if self.image_projection:
            self.image_projection.to(self.device)
            self.text_projection.to(self.device)
        self.criterion.to(self.device)
        self.hard_negative_criterion.to(self.device)

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_history = []

    # ---------------------------------------------------------------- #
    #  Projection head builder
    # ---------------------------------------------------------------- #

    def _build_projection_heads(self):
        cfg = self.config.finetune

        if cfg.use_improved_projection:
            logger.info(
                f"Using ImprovedProjectionHead: "
                f"{self.clip_embedding_dim} -> {cfg.projection_hidden_dim} "
                f"-> {cfg.projection_dim}  "
                f"(layers={cfg.projection_num_layers}, "
                f"dropout={cfg.projection_dropout})"
            )
            image_proj = ImprovedProjectionHead(
                input_dim=self.clip_embedding_dim,
                hidden_dim=cfg.projection_hidden_dim,
                output_dim=cfg.projection_dim,
                num_layers=cfg.projection_num_layers,
                dropout=cfg.projection_dropout,
            )
            text_proj = ImprovedProjectionHead(
                input_dim=self.clip_embedding_dim,
                hidden_dim=cfg.projection_hidden_dim,
                output_dim=cfg.projection_dim,
                num_layers=cfg.projection_num_layers,
                dropout=cfg.projection_dropout,
            )
        else:
            logger.info("Using original MedicalProjectionHead")
            image_proj = MedicalProjectionHead(
                self.clip_embedding_dim, cfg.projection_dim
            )
            text_proj = MedicalProjectionHead(
                self.clip_embedding_dim, cfg.projection_dim
            )

        return image_proj, text_proj

    # ---------------------------------------------------------------- #
    #  Freezing strategy
    # ---------------------------------------------------------------- #

    def _setup_freezing(self):
        for param in self.model.parameters():
            param.requires_grad = False

        if hasattr(self.model.vision_model, "encoder"):
            visual_layers = self.model.vision_model.encoder.layers
            num_visual = len(visual_layers)
            unfreeze_from = max(
                0, num_visual - self.config.finetune.unfreeze_visual_layers
            )
            for i in range(unfreeze_from, num_visual):
                for param in visual_layers[i].parameters():
                    param.requires_grad = True
            logger.info(
                f"Unfroze visual layers {unfreeze_from}-{num_visual-1}"
            )

        if hasattr(self.model.text_model, "encoder"):
            text_layers = self.model.text_model.encoder.layers
            num_text = len(text_layers)
            unfreeze_from = max(
                0, num_text - self.config.finetune.unfreeze_text_layers
            )
            for i in range(unfreeze_from, num_text):
                for param in text_layers[i].parameters():
                    param.requires_grad = True
            logger.info(
                f"Unfroze text layers {unfreeze_from}-{num_text-1}"
            )

        if hasattr(self.model, "visual_projection"):
            for param in self.model.visual_projection.parameters():
                param.requires_grad = True
        if hasattr(self.model, "text_projection"):
            for param in self.model.text_projection.parameters():
                param.requires_grad = True

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            f"Params -- total: {total:,} | trainable: {trainable:,} "
            f"({100*trainable/total:.1f}%)"
        )

    # ---------------------------------------------------------------- #
    #  Robust embedding extraction
    # ---------------------------------------------------------------- #

    def _extract_image_features(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            pooled = vision_outputs.pooler_output
        elif hasattr(vision_outputs, "last_hidden_state"):
            pooled = vision_outputs.last_hidden_state[:, 0, :]
        elif isinstance(vision_outputs, (tuple, list)):
            pooled = vision_outputs[1] if len(vision_outputs) > 1 else vision_outputs[0][:, 0, :]
        else:
            raise ValueError(
                f"Unexpected vision output type: {type(vision_outputs)}"
            )

        if not isinstance(pooled, torch.Tensor):
            raise TypeError(
                f"Expected tensor from vision model, got {type(pooled)}"
            )

        if hasattr(self.model, "visual_projection") and self.model.visual_projection is not None:
            image_features = self.model.visual_projection(pooled)
        else:
            image_features = pooled

        return image_features

    def _extract_text_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
            pooled = text_outputs.pooler_output
        elif hasattr(text_outputs, "last_hidden_state"):
            if input_ids is not None:
                eos_positions = input_ids.argmax(dim=-1)
                pooled = text_outputs.last_hidden_state[
                    torch.arange(
                        text_outputs.last_hidden_state.shape[0],
                        device=input_ids.device,
                    ),
                    eos_positions,
                ]
            else:
                pooled = text_outputs.last_hidden_state[:, -1, :]
        elif isinstance(text_outputs, (tuple, list)):
            pooled = text_outputs[1] if len(text_outputs) > 1 else text_outputs[0][:, -1, :]
        else:
            raise ValueError(
                f"Unexpected text output type: {type(text_outputs)}"
            )

        if not isinstance(pooled, torch.Tensor):
            raise TypeError(
                f"Expected tensor from text model, got {type(pooled)}"
            )

        if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
            text_features = self.model.text_projection(pooled)
        else:
            text_features = pooled

        return text_features

    def _get_embeddings(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embeds = self._extract_image_features(pixel_values)
        text_embeds = self._extract_text_features(input_ids, attention_mask)

        if self.image_projection and self.text_projection:
            image_embeds = self.image_projection(image_embeds)
            text_embeds = self.text_projection(text_embeds)

        return image_embeds, text_embeds

    # ---------------------------------------------------------------- #
    #  Public encode methods
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if self.image_projection:
            self.image_projection.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        image_features = self._extract_image_features(image)

        if self.image_projection:
            image_features = self.image_projection(image_features)

        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    @torch.no_grad()
    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.model.eval()
        if self.text_projection:
            self.text_projection.eval()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        text_features = self._extract_text_features(input_ids, attention_mask)

        if self.text_projection:
            text_features = self.text_projection(text_features)

        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

    @torch.no_grad()
    def encode_text_from_string(self, text: str) -> torch.Tensor:
        """
        NEW: Encode a raw text string into an embedding.
        Used for multi-modal search and text-based queries.

        Args:
            text: Raw text string (e.g., "bilateral pleural effusion")

        Returns:
            Normalized text embedding tensor (1, D)
        """
        self.model.eval()
        if self.text_projection:
            self.text_projection.eval()

        inputs = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        text_features = self._extract_text_features(input_ids, attention_mask)

        if self.text_projection:
            text_features = self.text_projection(text_features)

        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

    # ---------------------------------------------------------------- #
    #  Optimizer setup
    # ---------------------------------------------------------------- #

    def _get_optimizer_and_scheduler(
        self, num_training_steps: int
    ) -> Tuple:
        param_groups = []

        clip_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        if clip_params:
            param_groups.append({
                "params": clip_params,
                "lr": self.config.finetune.learning_rate,
                "weight_decay": self.config.finetune.weight_decay,
            })

        if self.image_projection:
            param_groups.append({
                "params": self.image_projection.parameters(),
                "lr": self.config.finetune.learning_rate * 10,
                "weight_decay": self.config.finetune.weight_decay,
            })
            param_groups.append({
                "params": self.text_projection.parameters(),
                "lr": self.config.finetune.learning_rate * 10,
                "weight_decay": self.config.finetune.weight_decay,
            })

        param_groups.append({
            "params": [self.criterion.temperature],
            "lr": self.config.finetune.learning_rate * 5,
            "weight_decay": 0.0,
        })

        optimizer = AdamW(param_groups)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.finetune.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, scheduler

    def _all_trainable_params(self) -> List[torch.Tensor]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.image_projection:
            params += list(self.image_projection.parameters())
            params += list(self.text_projection.parameters())
        return params

    # ---------------------------------------------------------------- #
    #  Training step with optional hard negatives
    # ---------------------------------------------------------------- #

    def _training_step(
        self,
        batch: Dict,
        use_hard_negatives: bool = False,
    ) -> Dict[str, torch.Tensor]:
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        image_embeds, text_embeds = self._get_embeddings(
            pixel_values, input_ids, attention_mask
        )

        standard_loss_dict = self.criterion(image_embeds, text_embeds)

        result = {
            "loss": standard_loss_dict["loss"],
            "standard_loss": standard_loss_dict["loss"],
            "i2t_accuracy": standard_loss_dict["i2t_accuracy"],
            "t2i_accuracy": standard_loss_dict["t2i_accuracy"],
            "logit_scale": standard_loss_dict["logit_scale"],
        }

        if (
            use_hard_negatives
            and "hard_negative" in batch
            and self.config.finetune.loss_type in ("hard_negative", "combined")
        ):
            hn_pixel_values = batch["hard_negative"]["pixel_values"].to(
                self.device
            )

            hn_image_embeds = self._extract_image_features(hn_pixel_values)
            if self.image_projection:
                hn_image_embeds = self.image_projection(hn_image_embeds)

            hn_loss_dict = self.hard_negative_criterion(
                anchor=image_embeds,
                positive=text_embeds,
                hard_negatives=hn_image_embeds,
            )

            hn_weight = self.config.finetune.hard_negative_weight
            combined_loss = (
                (1.0 - hn_weight) * standard_loss_dict["loss"]
                + hn_weight * hn_loss_dict["loss"]
            )

            result["loss"] = combined_loss
            result["hard_negative_loss"] = hn_loss_dict["loss"]
            result["hn_accuracy"] = hn_loss_dict["accuracy"]
            result["avg_pos_sim"] = hn_loss_dict["avg_pos_sim"]
            result["avg_neg_sim"] = hn_loss_dict["avg_neg_sim"]

        return result

    # ---------------------------------------------------------------- #
    #  Main training loop
    # ---------------------------------------------------------------- #

    def train(
        self,
        train_loader,
        val_loader=None,
        use_hard_negatives: bool = False,
    ):
        num_training_steps = (
            len(train_loader)
            * self.config.finetune.num_epochs
            // self.config.finetune.gradient_accumulation_steps
        )
        optimizer, scheduler = self._get_optimizer_and_scheduler(
            num_training_steps
        )

        scaler = GradScaler() if self.config.finetune.fp16 else None

        logger.info("=" * 60)
        logger.info("Starting CLIP Fine-Tuning on MIMIC-CXR")
        logger.info(f"  Epochs: {self.config.finetune.num_epochs}")
        logger.info(f"  Batch size: {self.config.finetune.batch_size}")
        logger.info(f"  Total steps: {num_training_steps}")
        logger.info(f"  FP16: {self.config.finetune.fp16}")
        logger.info(f"  Hard negatives: {use_hard_negatives}")
        logger.info(f"  Loss type: {self.config.finetune.loss_type}")
        logger.info(
            f"  Projection: "
            f"{'improved' if self.config.finetune.use_improved_projection else 'standard'}"
        )
        logger.info("=" * 60)

        for epoch in range(self.config.finetune.num_epochs):
            self.model.train()
            if self.image_projection:
                self.image_projection.train()
                self.text_projection.train()

            epoch_losses = []
            epoch_i2t_acc = []
            epoch_t2i_acc = []
            epoch_hn_losses = []

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.config.finetune.num_epochs}",
            )

            for step, batch in enumerate(pbar):
                if self.config.finetune.fp16:
                    with autocast():
                        loss_dict = self._training_step(
                            batch, use_hard_negatives=use_hard_negatives
                        )
                        loss = (
                            loss_dict["loss"]
                            / self.config.finetune.gradient_accumulation_steps
                        )

                    scaler.scale(loss).backward()

                    if (step + 1) % self.config.finetune.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self._all_trainable_params(),
                            self.config.finetune.max_grad_norm,
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                        self.global_step += 1
                else:
                    loss_dict = self._training_step(
                        batch, use_hard_negatives=use_hard_negatives
                    )
                    loss = (
                        loss_dict["loss"]
                        / self.config.finetune.gradient_accumulation_steps
                    )

                    loss.backward()

                    if (step + 1) % self.config.finetune.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self._all_trainable_params(),
                            self.config.finetune.max_grad_norm,
                        )
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        self.global_step += 1

                epoch_losses.append(loss_dict["loss"].item())
                epoch_i2t_acc.append(loss_dict["i2t_accuracy"].item())
                epoch_t2i_acc.append(loss_dict["t2i_accuracy"].item())

                if "hard_negative_loss" in loss_dict:
                    epoch_hn_losses.append(
                        loss_dict["hard_negative_loss"].item()
                    )

                postfix = {
                    "loss": f"{np.mean(epoch_losses[-50:]):.4f}",
                    "i2t": f"{np.mean(epoch_i2t_acc[-50:]):.3f}",
                    "t2i": f"{np.mean(epoch_t2i_acc[-50:]):.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
                if epoch_hn_losses:
                    postfix["hn"] = f"{np.mean(epoch_hn_losses[-50:]):.4f}"
                pbar.set_postfix(postfix)

                if (
                    val_loader
                    and self.global_step > 0
                    and self.global_step
                    % self.config.finetune.eval_every_n_steps
                    == 0
                ):
                    val_metrics = self.evaluate(val_loader)
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Val I2T: {val_metrics['i2t_accuracy']:.3f} | "
                        f"Val T2I: {val_metrics['t2i_accuracy']:.3f}"
                    )
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self.save_checkpoint("best_model")
                        logger.info("* New best model saved!")

                    self.model.train()
                    if self.image_projection:
                        self.image_projection.train()
                        self.text_projection.train()

            # End-of-epoch
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(epoch_losses)),
                "train_i2t_acc": float(np.mean(epoch_i2t_acc)),
                "train_t2i_acc": float(np.mean(epoch_t2i_acc)),
            }

            if epoch_hn_losses:
                epoch_metrics["train_hn_loss"] = float(
                    np.mean(epoch_hn_losses)
                )

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                epoch_metrics.update({
                    "val_loss": val_metrics["loss"],
                    "val_i2t_acc": val_metrics["i2t_accuracy"],
                    "val_t2i_acc": val_metrics["t2i_accuracy"],
                })

            self.training_history.append(epoch_metrics)
            logger.info(f"Epoch {epoch+1}: {epoch_metrics}")

            if (epoch + 1) % self.config.finetune.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")

        self.save_checkpoint("final_model")
        self._save_training_history()
        logger.info("Training complete!")
        return self.training_history

    @torch.no_grad()
    def evaluate(self, val_loader) -> Dict[str, float]:
        self.model.eval()
        if self.image_projection:
            self.image_projection.eval()
            self.text_projection.eval()

        all_losses, all_i2t, all_t2i = [], [], []

        for batch in tqdm(val_loader, desc="Eval", leave=False):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            image_embeds, text_embeds = self._get_embeddings(
                pixel_values, input_ids, attention_mask
            )
            loss_dict = self.criterion(image_embeds, text_embeds)

            all_losses.append(loss_dict["loss"].item())
            all_i2t.append(loss_dict["i2t_accuracy"].item())
            all_t2i.append(loss_dict["t2i_accuracy"].item())

        return {
            "loss": float(np.mean(all_losses)),
            "i2t_accuracy": float(np.mean(all_i2t)),
            "t2i_accuracy": float(np.mean(all_t2i)),
        }

    # ---------------------------------------------------------------- #
    #  Checkpoint save / load
    # ---------------------------------------------------------------- #

    def save_checkpoint(self, name: str):
        save_path = os.path.join(self.config.checkpoint_dir, name)
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

        if self.image_projection:
            torch.save(
                self.image_projection.state_dict(),
                os.path.join(save_path, "image_projection.pt"),
            )
            torch.save(
                self.text_projection.state_dict(),
                os.path.join(save_path, "text_projection.pt"),
            )

        torch.save(
            self.criterion.state_dict(),
            os.path.join(save_path, "criterion.pt"),
        )

        # Save projection config so we know dimensions when loading
        proj_config = {
            "use_projection_head": self.config.finetune.use_projection_head,
            "use_improved_projection": self.config.finetune.use_improved_projection,
            "projection_dim": self.config.finetune.projection_dim,
            "projection_hidden_dim": self.config.finetune.projection_hidden_dim,
            "projection_num_layers": self.config.finetune.projection_num_layers,
            "projection_dropout": self.config.finetune.projection_dropout,
            "clip_embedding_dim": self.clip_embedding_dim,
        }
        with open(os.path.join(save_path, "projection_config.json"), "w") as f:
            json.dump(proj_config, f)

        logger.info(f"Checkpoint saved: {save_path}")

    def load_checkpoint(self, name: str):
        load_path = os.path.join(self.config.checkpoint_dir, name)

        if not os.path.exists(load_path):
            logger.warning(f"Checkpoint {load_path} not found!")
            return

        self.model = CLIPModel.from_pretrained(load_path)
        self.processor = CLIPProcessor.from_pretrained(load_path)
        self.model.to(self.device)

        self.clip_embedding_dim = self.model.config.projection_dim
        logger.info(f"Loaded CLIP, projection_dim={self.clip_embedding_dim}")

        proj_config_path = os.path.join(load_path, "projection_config.json")
        img_proj_path = os.path.join(load_path, "image_projection.pt")

        if os.path.exists(img_proj_path):
            if os.path.exists(proj_config_path):
                with open(proj_config_path) as f:
                    proj_cfg = json.load(f)
                proj_dim = proj_cfg.get(
                    "projection_dim",
                    self.config.finetune.projection_dim,
                )
                embed_dim = proj_cfg.get(
                    "clip_embedding_dim",
                    self.clip_embedding_dim,
                )
                use_improved = proj_cfg.get(
                    "use_improved_projection",
                    self.config.finetune.use_improved_projection,
                )
                hidden_dim = proj_cfg.get(
                    "projection_hidden_dim",
                    self.config.finetune.projection_hidden_dim,
                )
                num_layers = proj_cfg.get(
                    "projection_num_layers",
                    self.config.finetune.projection_num_layers,
                )
                dropout = proj_cfg.get(
                    "projection_dropout",
                    self.config.finetune.projection_dropout,
                )
            else:
                proj_dim = self.config.finetune.projection_dim
                embed_dim = self.clip_embedding_dim
                use_improved = False
                hidden_dim = self.config.finetune.projection_hidden_dim
                num_layers = self.config.finetune.projection_num_layers
                dropout = self.config.finetune.projection_dropout

            if use_improved:
                self.image_projection = ImprovedProjectionHead(
                    input_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    output_dim=proj_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                ).to(self.device)
                self.text_projection = ImprovedProjectionHead(
                    input_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    output_dim=proj_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                ).to(self.device)
            else:
                self.image_projection = MedicalProjectionHead(
                    embed_dim, proj_dim
                ).to(self.device)
                self.text_projection = MedicalProjectionHead(
                    embed_dim, proj_dim
                ).to(self.device)

            self.image_projection.load_state_dict(
                torch.load(img_proj_path, map_location=self.device)
            )
            self.text_projection.load_state_dict(
                torch.load(
                    os.path.join(load_path, "text_projection.pt"),
                    map_location=self.device,
                )
            )
            logger.info(
                f"Loaded projection heads: {embed_dim} -> {proj_dim} "
                f"(improved={use_improved})"
            )

        logger.info(f"Checkpoint loaded from {load_path}")

    def _save_training_history(self):
        path = os.path.join(self.config.output_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved: {path}")