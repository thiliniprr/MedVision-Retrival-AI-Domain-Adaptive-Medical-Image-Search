# main_pipeline.py
"""
End-to-end pipeline supporting 4 output types:
  1. Local VLM (TinyLLaVA) report — vlm_few_shot
  2. Ollama (llava-llama3:8b) report — ollama_few_shot
  3. Template report with weighted probabilities — template
  4. Visual gallery (top 3-5 side-by-side) — visual
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional
import logging
import json
from datetime import datetime

os.environ['HF_HOME'] = 'cache/huggingface'

from config import (
    PipelineConfig, ModelConfig, DatasetConfig,
    FineTuneConfig, FAISSConfig, RetrievalConfig,
    VLMConfig, OllamaVLMConfig,
)
from dataset_loader import DataLoaderFactory, MIMICCXRDataset
from clip_finetuner import CLIPFineTuner
from faiss_index_builder import FAISSIndexBuilder
from retrieval_engine import RetrievalEngine
from report_generator import ReportGenerator
from visual_report import VisualReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class MedicalImageRetrievalPipeline:
    """
    Master pipeline for medical image retrieval and report generation.

    4 output types:
      1. vlm_few_shot    — TinyLLaVA local VLM report
      2. ollama_few_shot — Ollama llava-llama3:8b report
      3. template        — Template report with weighted probabilities
      4. visual          — Side-by-side gallery saved to output folder
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Pipeline device: {self.device}")
        logger.info(f"VLM backend: {self.config.vlm_backend}")
        if self.config.vlm_backend == "ollama":
            logger.info(
                f"Ollama: {self.config.ollama_vlm.host} | "
                f"Model: {self.config.ollama_vlm.model_name}"
            )

        self.finetuner: Optional[CLIPFineTuner] = None
        self.index_builder: Optional[FAISSIndexBuilder] = None
        self.retrieval_engine: Optional[RetrievalEngine] = None
        self.report_generator: Optional[ReportGenerator] = None
        self.data_factory: Optional[DataLoaderFactory] = None
        self.train_dataset: Optional[MIMICCXRDataset] = None

    # ------------------------------------------------------------- #
    #  Initialisation helpers
    # ------------------------------------------------------------- #

    def _init_finetuner(self) -> CLIPFineTuner:
        if self.finetuner is None:
            self.finetuner = CLIPFineTuner(self.config)
        return self.finetuner

    def _init_data_factory(self) -> DataLoaderFactory:
        ft = self._init_finetuner()
        if self.data_factory is None:
            self.data_factory = DataLoaderFactory(
                self.config, ft.processor
            )
        return self.data_factory

    def _init_index_builder(self) -> FAISSIndexBuilder:
        ft = self._init_finetuner()
        if self.index_builder is None:
            self.index_builder = FAISSIndexBuilder(self.config, ft)
        return self.index_builder

    def _init_retrieval_engine(self) -> RetrievalEngine:
        ft = self._init_finetuner()
        ib = self._init_index_builder()
        if self.retrieval_engine is None:
            self.retrieval_engine = RetrievalEngine(
                self.config, ft, ib
            )
        return self.retrieval_engine

    def _init_report_generator(self) -> ReportGenerator:
        if self.report_generator is None:
            self.report_generator = ReportGenerator(self.config)
            try:
                df = self._init_data_factory()
                ds = df.get_full_dataset_for_indexing()
                self.report_generator.set_vlm_dataset(ds)
            except Exception:
                pass
        return self.report_generator

    def _ensure_ready_for_query(self):
        #self._init_finetuner()
        ib = self._init_index_builder()
        if ib.index is None:
            raise RuntimeError(
                "FAISS index not loaded. "
                "Call load_index() or build_index() first."
            )
        self._init_retrieval_engine()
        self._init_report_generator()

    # ------------------------------------------------------------- #
    #  Stage 1: Fine-tune CLIP
    # ------------------------------------------------------------- #

    def finetune(self):
        logger.info("=" * 60)
        logger.info("STAGE 1 -- Fine-tuning CLIP on MIMIC-CXR")
        logger.info("=" * 60)

        ft = self._init_finetuner()
        df = self._init_data_factory()
        ib = self._init_index_builder()

        train_loader, val_loader = df.get_dataloaders()
        logger.info(
            f"Train batches: {len(train_loader)} | "
            f"Val batches: {len(val_loader) if val_loader else 0}"
        )

        use_hn = self.config.finetune.use_hard_negatives
        mining_start = self.config.finetune.mining_start_epoch

        if not use_hn:
            logger.info("Training without hard negatives")
            history = ft.train(
                train_loader, val_loader,
                use_hard_negatives=False,
            )
        else:
            logger.info(
                f"Training WITH hard negatives "
                f"(mining starts at epoch {mining_start})"
            )

            total_epochs = self.config.finetune.num_epochs

            warmup_epochs = min(mining_start, total_epochs)
            if warmup_epochs > 0:
                logger.info(
                    f"Phase 1: Warm-up ({warmup_epochs} epochs)"
                )
                self.config.finetune.num_epochs = warmup_epochs
                ft.train(
                    train_loader, val_loader,
                    use_hard_negatives=False,
                )

            remaining_epochs = total_epochs - warmup_epochs
            if remaining_epochs > 0:
                logger.info(
                    f"Phase 2: Hard negative training "
                    f"({remaining_epochs} epochs)"
                )

                train_ds, _ = df.load_datasets()
                self.train_dataset = train_ds
                index_dataset = df.get_full_dataset_for_indexing()

                for hn_epoch in range(remaining_epochs):
                    actual_epoch = warmup_epochs + hn_epoch + 1
                    logger.info(
                        f"--- HN Epoch "
                        f"{actual_epoch}/{total_epochs} ---"
                    )

                    logger.info("Rebuilding FAISS index...")
                    ib.build_index(
                        index_dataset,
                        batch_size=self.config.finetune.batch_size,
                    )

                    re_ = self._init_retrieval_engine()
                    hard_neg_map = (
                        re_.mine_hard_negatives_for_training(
                            num_negatives=(
                                self.config.finetune
                                .num_hard_negatives
                            ),
                            search_k=(
                                self.config.finetune
                                .hard_negative_search_k
                            ),
                        )
                    )

                    train_ds.enable_hard_negative_sampling(
                        hard_neg_map
                    )

                    hn_train_loader = torch.utils.data.DataLoader(
                        train_ds,
                        batch_size=self.config.finetune.batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        collate_fn=DataLoaderFactory.collate_fn,
                        drop_last=True,
                    )

                    self.config.finetune.num_epochs = 1
                    ft.train(
                        hn_train_loader, val_loader,
                        use_hard_negatives=True,
                    )

                train_ds.disable_hard_negative_sampling()

            self.config.finetune.num_epochs = total_epochs
            history = ft.training_history

        logger.info("Fine-tuning complete.")
        return history

    # ------------------------------------------------------------- #
    #  Stage 2: Build FAISS index
    # ------------------------------------------------------------- #

    def build_index(self):
        logger.info("=" * 60)
        logger.info("STAGE 2 -- Building FAISS index")
        logger.info("=" * 60)

        ft = self._init_finetuner()
        ib = self._init_index_builder()
        df = self._init_data_factory()

        dataset = df.get_full_dataset_for_indexing()
        ib.build_index(
            dataset, batch_size=self.config.finetune.batch_size
        )
        ib.save_index()

        stats = ib.get_index_stats()
        logger.info(f"Index stats: {json.dumps(stats, indent=2)}")
        return stats

    # ------------------------------------------------------------- #
    #  Stage 3: Query + Report (unified for all 4 output types)
    # ------------------------------------------------------------- #

    def query(
        self,
        image_path: str,
        top_k: Optional[int] = None,
        report_method: Optional[str] = None,
        text_query: Optional[str] = None,
        use_query_expansion: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        use_multimodal: Optional[bool] = None,
    ) -> Dict:
        """
        Retrieve similar cases and generate a report.

        report_method options (4 main output types):
          "vlm_few_shot"    — Type 1: Local TinyLLaVA report
          "ollama_few_shot" — Type 2: Ollama llava-llama3:8b report
          "template"        — Type 3: Template with weighted probs
          "visual"          — Type 4: Side-by-side gallery

        Additional methods:
          "vlm_zero_shot", "ollama_zero_shot"
          "weighted", "majority", "concat"
        """
        logger.info("=" * 60)
        logger.info("STAGE 3 -- Query & Report Generation")
        logger.info("=" * 60)

        self._ensure_ready_for_query()

        re_ = self._init_retrieval_engine()
        rg = self._init_report_generator()

        image = Image.open(image_path).convert("RGB")
        logger.info(f"Query image: {image_path} ({image.size})")

        results = re_.retrieve_similar(
            image,
            top_k=top_k,
            text_query=text_query,
            use_query_expansion=use_query_expansion,
            use_reranking=use_reranking,
            use_multimodal=use_multimodal,
        )
        logger.info(f"Retrieved {len(results)} similar cases")

        # Generate the report
        report_output = rg.generate_report(
            results,
            method=report_method,
            query_image=image,
        )

        # ── Type 4 special handling: save visual gallery ──
        if report_method == "visual" or report_output.get("save_gallery"):
            self._save_visual_gallery(
                query_image=image,
                retrieval_results=results,
                detected_conditions=report_output.get(
                    "detected_conditions"
                ),
                output_dir=self.config.output_dir,
            )

        logger.info("\n" + report_output["report"])
        return report_output

    def query_pil(
        self,
        image: Image.Image,
        top_k: Optional[int] = None,
        report_method: Optional[str] = None,
        text_query: Optional[str] = None,
        use_query_expansion: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        use_multimodal: Optional[bool] = None,
    ) -> Dict:
        """Same as query() but accepts a PIL Image."""
        self._ensure_ready_for_query()

        re_ = self._init_retrieval_engine()
        rg = self._init_report_generator()

        image = image.convert("RGB")
        results = re_.retrieve_similar(
            image,
            top_k=top_k,
            text_query=text_query,
            use_query_expansion=use_query_expansion,
            use_reranking=use_reranking,
            use_multimodal=use_multimodal,
        )

        report_output = rg.generate_report(
            results,
            method=report_method,
            query_image=image,
        )

        # Save gallery for visual method
        if report_method == "visual" or report_output.get("save_gallery"):
            self._save_visual_gallery(
                query_image=image,
                retrieval_results=results,
                detected_conditions=report_output.get(
                    "detected_conditions"
                ),
                output_dir=self.config.output_dir,
            )

        return report_output

    # ------------------------------------------------------------- #
    #  Visual gallery save helper
    # ------------------------------------------------------------- #

    def _save_visual_gallery(
        self,
        query_image: Image.Image,
        retrieval_results: list,
        detected_conditions: Optional[Dict] = None,
        output_dir: str = "./output",
    ):
        """
        Create and save visual gallery images to the output folder.
        Produces:
          - visual_gallery.png  (side-by-side comparison)
          - detailed_visual_report.png (with conditions panel)
        """
        visual_gen = VisualReportGenerator()

        # Try to set dataset for image loading
        try:
            df = self._init_data_factory()
            ds = df.get_full_dataset_for_indexing()
            visual_gen.set_dataset(ds)
        except Exception:
            logger.warning(
                "Dataset not available for visual gallery images"
            )

        os.makedirs(output_dir, exist_ok=True)

        # Gallery image
        gallery_path = visual_gen.create_visual_report(
            query_image=query_image,
            retrieval_results=retrieval_results,
            save_path=os.path.join(output_dir, "visual_gallery.png"),
            output_dir=output_dir,
        )

        # Detailed report image
        detailed_path = visual_gen.create_detailed_visual_report(
            query_image=query_image,
            retrieval_results=retrieval_results,
            detected_conditions=detected_conditions,
            save_path=os.path.join(
                output_dir, "detailed_visual_report.png"
            ),
            output_dir=output_dir,
        )

        if gallery_path:
            logger.info(f"Gallery saved: {gallery_path}")
        if detailed_path:
            logger.info(f"Detailed report saved: {detailed_path}")

        return gallery_path, detailed_path

    # ------------------------------------------------------------- #
    #  Convenience methods for each output type
    # ------------------------------------------------------------- #

    def query_vlm(
        self,
        image_path: str,
        top_k: Optional[int] = None,
        mode: str = "few_shot",
        text_query: Optional[str] = None,
        use_query_expansion: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
    ) -> Dict:
        """
        Type 1: Query with local TinyLLaVA VLM report.
        Uses query image + k-retrieved similar images/captions.
        """
        method = f"vlm_{mode}"
        return self.query(
            image_path=image_path,
            top_k=top_k,
            report_method=method,
            text_query=text_query,
            use_query_expansion=use_query_expansion,
            use_reranking=use_reranking,
        )

    def query_ollama(
        self,
        image_path: str,
        top_k: Optional[int] = None,
        mode: str = "few_shot",
        text_query: Optional[str] = None,
        use_query_expansion: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
    ) -> Dict:
        """
        Type 2: Query with Ollama llava-llama3:8b report.
        Uses query image + k-retrieved similar images/captions.
        """
        method = f"ollama_{mode}"
        return self.query(
            image_path=image_path,
            top_k=top_k,
            report_method=method,
            text_query=text_query,
            use_query_expansion=use_query_expansion,
            use_reranking=use_reranking,
        )

    def query_template(
        self,
        image_path: str,
        top_k: Optional[int] = None,
        text_query: Optional[str] = None,
    ) -> Dict:
        """
        Type 3: Template report with weighted probability scores.
        Uses top 1-3 retrieved images with score >= 0.3.
        """
        return self.query(
            image_path=image_path,
            top_k=top_k or 3,
            report_method="template",
            text_query=text_query,
        )

    def query_visual(
        self,
        image_path: str,
        top_k: Optional[int] = None,
        text_query: Optional[str] = None,
    ) -> Dict:
        """
        Type 4: Side-by-side visual gallery of top 3-5 similar images.
        Gallery images saved to the output folder.
        """
        return self.query(
            image_path=image_path,
            top_k=top_k or 5,
            report_method="visual",
            text_query=text_query,
        )

    # PIL variants
    def query_vlm_pil(
        self,
        image: Image.Image,
        top_k: Optional[int] = None,
        mode: str = "few_shot",
        text_query: Optional[str] = None,
    ) -> Dict:
        """Type 1 with PIL Image input."""
        method = f"vlm_{mode}"
        return self.query_pil(
            image=image, top_k=top_k,
            report_method=method, text_query=text_query,
        )

    def query_ollama_pil(
        self,
        image: Image.Image,
        top_k: Optional[int] = None,
        mode: str = "few_shot",
        text_query: Optional[str] = None,
    ) -> Dict:
        """Type 2 with PIL Image input."""
        method = f"ollama_{mode}"
        return self.query_pil(
            image=image, top_k=top_k,
            report_method=method, text_query=text_query,
        )

    def query_template_pil(
        self,
        image: Image.Image,
        top_k: Optional[int] = None,
        text_query: Optional[str] = None,
    ) -> Dict:
        """Type 3 with PIL Image input."""
        return self.query_pil(
            image=image, top_k=top_k or 3,
            report_method="template", text_query=text_query,
        )

    def query_visual_pil(
        self,
        image: Image.Image,
        top_k: Optional[int] = None,
        text_query: Optional[str] = None,
    ) -> Dict:
        """Type 4 with PIL Image input."""
        return self.query_pil(
            image=image, top_k=top_k or 5,
            report_method="visual", text_query=text_query,
        )

    # ------------------------------------------------------------- #
    #  Generate all 4 output types at once
    # ------------------------------------------------------------- #

    def query_all_types(
        self,
        image_path: str,
        top_k: int = 5,
        text_query: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Run all 4 output types for a single query image.

        Returns:
            Dict with keys: "vlm", "ollama", "template", "visual"
        """
        logger.info("=" * 60)
        logger.info("Generating ALL 4 output types")
        logger.info("=" * 60)

        self._ensure_ready_for_query()

        re_ = self._init_retrieval_engine()
        rg = self._init_report_generator()

        image = Image.open(image_path).convert("RGB")

        # Retrieve once, reuse for all methods
        results = re_.retrieve_similar(
            image, top_k=top_k, text_query=text_query,
        )
        logger.info(f"Retrieved {len(results)} similar cases")

        all_outputs = {}

        # Type 1: Local VLM (TinyLLaVA)
        logger.info("─── Type 1: Local VLM (TinyLLaVA) ───")
        try:
            all_outputs["vlm"] = rg.generate_report(
                results, method="vlm_few_shot", query_image=image,
            )
        except Exception as e:
            logger.error(f"VLM report failed: {e}")
            all_outputs["vlm"] = {
                "report": f"VLM report failed: {e}",
                "method": "vlm_few_shot",
                "success": False,
                "error": str(e),
            }

        # Type 2: Ollama (llava-llama3:8b)
        logger.info("─── Type 2: Ollama (llava-llama3:8b) ───")
        try:
            all_outputs["ollama"] = rg.generate_report(
                results, method="ollama_few_shot", query_image=image,
            )
        except Exception as e:
            logger.error(f"Ollama report failed: {e}")
            all_outputs["ollama"] = {
                "report": f"Ollama report failed: {e}",
                "method": "ollama_few_shot",
                "success": False,
                "error": str(e),
            }

        # Type 3: Template with weighted probabilities
        logger.info("─── Type 3: Template Report ───")
        try:
            all_outputs["template"] = rg.generate_report(
                results, method="template", query_image=image,
            )
        except Exception as e:
            logger.error(f"Template report failed: {e}")
            all_outputs["template"] = {
                "report": f"Template report failed: {e}",
                "method": "template",
                "success": False,
                "error": str(e),
            }

        # Type 4: Visual gallery
        logger.info("─── Type 4: Visual Gallery ───")
        try:
            all_outputs["visual"] = rg.generate_report(
                results, method="visual", query_image=image,
            )
            # Save gallery images
            self._save_visual_gallery(
                query_image=image,
                retrieval_results=results,
                detected_conditions=all_outputs["visual"].get(
                    "detected_conditions"
                ),
                output_dir=self.config.output_dir,
            )
        except Exception as e:
            logger.error(f"Visual report failed: {e}")
            all_outputs["visual"] = {
                "report": f"Visual report failed: {e}",
                "method": "visual",
                "success": False,
                "error": str(e),
            }

        # Print summaries
        for type_name, output in all_outputs.items():
            status = "✅" if output.get("success") else "❌"
            logger.info(
                f"{status} Type '{type_name}' — "
                f"method={output.get('method', '?')}"
            )

        return all_outputs

    # ------------------------------------------------------------- #
    #  Ollama connection check
    # ------------------------------------------------------------- #

    def check_ollama_connection(self) -> Dict:
        """Check connection to Ollama server and model status."""
        from ollama_vlm_report_generator import OllamaClient

        client = OllamaClient(self.config.ollama_vlm)

        status = {
            "host": self.config.ollama_vlm.host,
            "target_model": self.config.ollama_vlm.model_name,
            "reachable": False,
            "available_models": [],
            "model_ready": False,
        }

        if client.health_check():
            status["reachable"] = True
            status["available_models"] = client.list_models()
            status["model_ready"] = client.model_available(
                self.config.ollama_vlm.model_name
            )
            if status["model_ready"]:
                info = client.get_model_info(
                    self.config.ollama_vlm.model_name
                )
                if info:
                    details = info.get("details", {})
                    status["model_info"] = {
                        "family": details.get("family", "unknown"),
                        "parameter_size": details.get(
                            "parameter_size", "unknown"
                        ),
                        "quantization_level": details.get(
                            "quantization_level", "unknown"
                        ),
                    }
        else:
            status["error"] = (
                f"Cannot reach Ollama at "
                f"{self.config.ollama_vlm.host}"
            )

        logger.info(
            f"Ollama status: {json.dumps(status, indent=2)}"
        )
        return status

    # ------------------------------------------------------------- #
    #  Model management
    # ------------------------------------------------------------- #

    def unload_vlm(self):
        """Free local VLM model memory."""
        rg = self._init_report_generator()
        try:
            vlm_gen = rg._get_vlm_generator()
            vlm_gen.unload_model()
            logger.info("Local VLM model unloaded")
        except Exception as e:
            logger.warning(f"Could not unload local VLM: {e}")

    # ------------------------------------------------------------- #
    #  Checkpoint management
    # ------------------------------------------------------------- #

    def load_checkpoint(self, name: str = "final_model"):
        ft = self._init_finetuner()
        ft.load_checkpoint(name)
        logger.info(f"Loaded checkpoint: {name}")

    def load_index(self):
        ib = self._init_index_builder()
        ib.load_index()
        logger.info("FAISS index loaded from disk.")

    # ------------------------------------------------------------- #
    #  Full pipeline
    # ------------------------------------------------------------- #

    def run_full_pipeline(
        self,
        query_image_path: Optional[str] = None,
        skip_finetune: bool = False,
        skip_index: bool = False,
        report_method: str = "template",
    ) -> Optional[Dict]:
        start = datetime.now()

        if skip_finetune:
            logger.info("Skipping fine-tuning, loading checkpoint...")
            self.load_checkpoint("final_model")
        else:
            self.finetune()

        if skip_index:
            logger.info("Skipping index build, loading from disk...")
            self.load_index()
        else:
            self.build_index()

        report = None
        if query_image_path and os.path.exists(query_image_path):
            report = self.query(
                query_image_path,
                report_method=report_method,
            )

        elapsed = datetime.now() - start
        logger.info(f"Total pipeline time: {elapsed}")

        return report


# ================================================================== #
#  CLI
# ================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Medical Image Retrieval & Report Generation\n\n"
            "4 Output Types:\n"
            "  1. vlm_few_shot    — Local TinyLLaVA report\n"
            "  2. ollama_few_shot — Ollama llava-llama3:8b report\n"
            "  3. template        — Weighted probability template\n"
            "  4. visual          — Side-by-side gallery\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    p.add_argument(
        "--mode",
        choices=[
            "full", "finetune", "index", "query",
            "query_vlm", "query_ollama", "query_template",
            "query_visual", "query_all",
            "check_ollama",
            "download_model",  # <-- ADD THIS
            "check_model",     # <-- ADD THIS
        ],
        default="full",
        help=(
            "Pipeline stage to run.\n"
            "  full            — entire pipeline\n"
            "  finetune        — CLIP fine-tuning only\n"
            "  index           — FAISS index only\n"
            "  query           — retrieve + report (any method)\n"
            "  query_vlm       — Type 1: TinyLLaVA report\n"
            "  query_ollama    — Type 2: Ollama report\n"
            "  query_template  — Type 3: template report\n"
            "  query_visual    — Type 4: visual gallery\n"
            "  query_all       — all 4 types at once\n"
            "  check_ollama    — test Ollama connection"
        ),
    )

    # General
    p.add_argument("--query_image", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default="final_model")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument(
        "--report_method",
        choices=[
            "weighted", "majority", "concat", "template", "visual",
            "vlm_few_shot", "vlm_zero_shot",
            "ollama_few_shot", "ollama_zero_shot",
        ],
        default="template",
        help="Report generation method.",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--skip_finetune", action="store_true")
    p.add_argument("--skip_index", action="store_true")
    p.add_argument(
        "--output_dir", type=str, default="./output",
        help="Directory for output files",
    )

    # ============================================================
    # Enhancement flags
    # ============================================================
    enhancement = p.add_argument_group("Enhancement flags")
    enhancement.add_argument(
        "--text_query", type=str, default=None,
        help="Optional text query for multi-modal search",
    )
    enhancement.add_argument(
        "--no_query_expansion", action="store_true",
    )
    enhancement.add_argument(
        "--no_reranking", action="store_true",
    )
    enhancement.add_argument(
        "--no_hard_negatives", action="store_true",
    )
    enhancement.add_argument(
        "--no_whitening", action="store_true",
    )
    enhancement.add_argument(
        "--use_multimodal", action="store_true",
    )
    enhancement.add_argument(
        "--simple_projection", action="store_true",
    )

    # ============================================================
    # Local VLM flags (MedGemma 4B-IT)
    # ============================================================
    local_vlm = p.add_argument_group(
        "Local VLM (MedGemma 4B-IT — Type 1)"
    )
    local_vlm.add_argument(
        "--vlm_model", type=str,
        default="google/medgemma-4b-it",
        help=(
            "Local VLM model name. Default: "
            "google/medgemma-4b-it (4B medical VLM)"
        ),
    )
    local_vlm.add_argument(
        "--vlm_4bit", action="store_true",
        help="Load local VLM in 4-bit quantization",
    )
    local_vlm.add_argument(
        "--vlm_8bit", action="store_true",
        help="Load local VLM in 8-bit quantization",
    )
    local_vlm.add_argument(
        "--hf_token", type=str, default=None,
        help=(
            "HuggingFace token for gated model access "
            "(MedGemma). Or set HF_TOKEN env variable."
        ),
    )
    local_vlm.add_argument(
        "--vlm_cache_dir", type=str,
        default="./cache/huggingface",
        help="Local directory to download/cache VLM model",
    )
    
    
    # ============================================================
    # VLM backend selection and shared settings
    # ============================================================
    vlm_general = p.add_argument_group(
        "VLM backend selection and shared settings"
    )
    vlm_general.add_argument(
        "--vlm_backend",
        choices=["local", "ollama"],
        default="ollama",
        help=(
            "VLM backend to use.\n"
            "  ollama — llava-llama3:8b on Ollama (default)\n"
            "  local  — TinyLLaVA model locally"
        ),
    )
    vlm_general.add_argument(
        "--vlm_mode", type=str,
        choices=["few_shot", "zero_shot"],
        default="few_shot",
        help="VLM generation mode",
    )
    vlm_general.add_argument(
        "--vlm_temperature", type=float, default=0.3,
        help="Temperature for VLM generation",
    )
    vlm_general.add_argument(
        "--vlm_max_tokens", type=int, default=512,
        help="Max new tokens for VLM generation",
    )
    vlm_general.add_argument(
        "--vlm_num_examples", type=int, default=3,
        help="Number of few-shot examples from retrieval",
    )
    vlm_general.add_argument(
        "--no_vlm", action="store_true",
        help="Disable VLM report generation entirely",
    )

    # ============================================================
    # Ollama settings (llava-llama3:8b — Type 2)
    # ============================================================
    ollama = p.add_argument_group(
        "Ollama remote VLM settings (llava-llama3:8b — Type 2)"
    )
    ollama.add_argument(
        "--ollama_host", type=str,
        default="http://86.50.170.53:11434",
        help="Ollama server URL.",
    )
    ollama.add_argument(
        "--ollama_model", type=str,
        default="llava-llama3:8b",
        help="Ollama model name.",
    )
    ollama.add_argument(
        "--ollama_timeout", type=int, default=180,
        help="Timeout in seconds for Ollama requests.",
    )
    ollama.add_argument(
        "--ollama_max_retries", type=int, default=3,
        help="Max retries for failed Ollama requests",
    )
    ollama.add_argument(
        "--ollama_stream", action="store_true",
        help="Use streaming for Ollama responses",
    )
    ollama.add_argument(
        "--ollama_keep_alive", type=str, default="15m",
        help="How long Ollama keeps model in memory.",
    )
    ollama.add_argument(
        "--ollama_seed", type=int, default=None,
        help="Random seed for reproducible generation",
    )
    ollama.add_argument(
        "--ollama_top_k", type=int, default=40,
        help="Top-K sampling for Ollama generation",
    )
    ollama.add_argument(
        "--ollama_top_p", type=float, default=0.9,
        help="Top-P (nucleus) sampling for Ollama generation",
    )
    ollama.add_argument(
        "--ollama_repeat_penalty", type=float, default=1.1,
        help="Repetition penalty for Ollama generation",
    )
    ollama.add_argument(
        "--ollama_image_max_size", type=int, default=1024,
        help="Max image dimension before sending to Ollama",
    )
    ollama.add_argument(
        "--ollama_image_quality", type=int, default=90,
        help="JPEG quality for image encoding (1-100)",
    )

    return p.parse_args()


def _build_config(args) -> PipelineConfig:
    """Build PipelineConfig from parsed CLI arguments."""

    config = PipelineConfig(
        model=ModelConfig(),
        dataset=DatasetConfig(
            max_train_samples=args.max_train_samples,
        ),
        finetune=FineTuneConfig(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            use_hard_negatives=not args.no_hard_negatives,
            use_improved_projection=not args.simple_projection,
        ),
        faiss=FAISSConfig(
            use_whitening=not args.no_whitening,
        ),
        retrieval=RetrievalConfig(
            top_k=args.top_k,
            use_query_expansion=not args.no_query_expansion,
            use_reranking=not args.no_reranking,
            use_multimodal_search=args.use_multimodal,
        ),
        vlm=VLMConfig(
            model_name=args.vlm_model,
            temperature=args.vlm_temperature,
            max_new_tokens=args.vlm_max_tokens,
            num_examples=args.vlm_num_examples,
            load_in_4bit=args.vlm_4bit,
            load_in_8bit=args.vlm_8bit,
            enabled=not args.no_vlm,
            hf_token=args.hf_token,
            local_model_dir=args.vlm_cache_dir,
        ),
        ollama_vlm=OllamaVLMConfig(
            host=args.ollama_host,
            model_name=args.ollama_model,
            temperature=args.vlm_temperature,
            max_new_tokens=args.vlm_max_tokens,
            num_examples=args.vlm_num_examples,
            top_p=args.ollama_top_p,
            top_k=args.ollama_top_k,
            repeat_penalty=args.ollama_repeat_penalty,
            timeout=args.ollama_timeout,
            max_retries=args.ollama_max_retries,
            stream=args.ollama_stream,
            keep_alive=args.ollama_keep_alive,
            seed=args.ollama_seed,
            image_max_size=args.ollama_image_max_size,
            image_quality=args.ollama_image_quality,
            enabled=not args.no_vlm,
        ),
        vlm_backend=args.vlm_backend,
        output_dir=args.output_dir,
        device=args.device,
    )

    return config


def _save_result(result: Dict, output_path: str):
    """Save query result to JSON."""
    serialisable = {}
    for k, v in result.items():
        if k == "similar_cases":
            serialisable["similar_cases"] = [
                {k2: v2 for k2, v2 in sc.items()}
                for sc in v
            ]
        else:
            serialisable[k] = v

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    logger.info(f"Result saved to {output_path}")


def _print_report_result(result: Dict):
    """Print report result with metadata."""
    print("\n" + result.get("report", "No report generated."))
    print()
    if result.get("success"):
        print(f"  Backend: {result.get('backend', 'unknown')}")
        print(f"  Model:   {result.get('model', 'unknown')}")
        if result.get("host"):
            print(f"  Host:    {result['host']}")
        print(f"  Examples: {result.get('num_examples', 0)}")
        print(f"  Method:  {result.get('method', 'unknown')}")
    else:
        print(f"  ERROR: {result.get('error', 'Unknown error')}")


def _print_ollama_status(status: Dict):
    """Pretty-print Ollama connection status."""
    print("\n" + "=" * 55)
    print("  OLLAMA CONNECTION CHECK")
    print("=" * 55)
    print(f"  Host:           {status['host']}")
    print(f"  Reachable:      {status['reachable']}")
    print(f"  Target model:   {status['target_model']}")
    print(f"  Model ready:    {status['model_ready']}")

    if status.get("available_models"):
        models_str = ", ".join(status["available_models"])
        print(f"  Available:      {models_str}")

    if status.get("model_info"):
        info = status["model_info"]
        print(f"  Family:         {info.get('family', '?')}")
        print(f"  Parameters:     {info.get('parameter_size', '?')}")
        print(f"  Quantization:   {info.get('quantization_level', '?')}")

    if "error" in status:
        print(f"\n  ERROR: {status['error']}")

    if status["reachable"] and not status["model_ready"]:
        print(
            f"\n  Model not found. Pull it on the server:"
            f"\n    ollama pull {status['target_model']}"
        )

    if not status["reachable"]:
        print(
            f"\n  Start Ollama on the server:"
            f"\n    OLLAMA_HOST=0.0.0.0:11434 ollama serve"
            f"\n"
            f"\n  Open firewall:"
            f"\n    sudo ufw allow 11434/tcp"
            f"\n"
            f"\n  Pull the model:"
            f"\n    ollama pull llava-llama3:8b"
        )

    print("=" * 55)


def _print_all_types_summary(all_outputs: Dict[str, Dict]):
    """Pretty-print summary for query_all mode."""
    print("\n" + "=" * 65)
    print("  ALL 4 OUTPUT TYPES — SUMMARY")
    print("=" * 65)

    type_labels = {
        "vlm": "Type 1: Local TinyLLaVA",
        "ollama": "Type 2: Ollama llava-llama3:8b",
        "template": "Type 3: Template (weighted probabilities)",
        "visual": "Type 4: Visual Gallery",
    }

    for key, label in type_labels.items():
        output = all_outputs.get(key, {})
        status = "✅ SUCCESS" if output.get("success") else "❌ FAILED"
        print(f"\n  {label}")
        print(f"  Status: {status}")

        if output.get("success"):
            method = output.get("method", "?")
            print(f"  Method: {method}")
            if output.get("backend"):
                print(f"  Backend: {output['backend']}")
            if output.get("model"):
                print(f"  Model: {output['model']}")
            if output.get("num_examples"):
                print(f"  Examples: {output['num_examples']}")
            if output.get("num_results"):
                print(f"  Results: {output['num_results']}")
        else:
            print(f"  Error: {output.get('error', 'Unknown')}")

    print("\n" + "=" * 65)

    # Print each report
    for key, label in type_labels.items():
        output = all_outputs.get(key, {})
        if output.get("success"):
            print(f"\n{'─' * 65}")
            print(f"  {label}")
            print(f"{'─' * 65}")
            print(output.get("report", "No report"))


def main():
    args = parse_args()
    config = _build_config(args)
    

    # ============================================================
    # Mode: check_ollama
    # ============================================================
    if args.mode == "check_ollama":
        pipeline = MedicalImageRetrievalPipeline(config)
        status = pipeline.check_ollama_connection()
        _print_ollama_status(status)
        sys.exit(0 if status["reachable"] else 1)

    pipeline = MedicalImageRetrievalPipeline(config)

    # ============================================================
    # Mode: full
    # ============================================================
    if args.mode == "full":
        pipeline.run_full_pipeline(
            query_image_path=args.query_image,
            skip_finetune=args.skip_finetune,
            skip_index=args.skip_index,
            report_method=args.report_method,
        )

    # ============================================================
    # Mode: finetune
    # ============================================================
    elif args.mode == "finetune":
        pipeline.finetune()

    # ============================================================
    # Mode: index
    # ============================================================
    elif args.mode == "index":
        pipeline.load_checkpoint(args.checkpoint)
        pipeline.build_index()

    # ============================================================
    # Mode: query (any report method)
    # ============================================================
    elif args.mode == "query":
        if not args.query_image:
            logger.error("--query_image is required for query mode.")
            sys.exit(1)

        pipeline.load_checkpoint(args.checkpoint)
        pipeline.load_index()

        result = pipeline.query(
            args.query_image,
            top_k=args.top_k,
            report_method=args.report_method,
            text_query=args.text_query,
        )

        output_path = os.path.join(
            config.output_dir, "query_result.json"
        )
        _save_result(result, output_path)
        _print_report_result(result)

    # ============================================================
    # Mode: query_vlm (Type 1: Local TinyLLaVA)
    # ============================================================
    elif args.mode == "query_vlm":
        if not args.query_image:
            logger.error("--query_image is required.")
            sys.exit(1)

        # ── FORCE local backend when using query_vlm mode ──
        config.vlm_backend = "local"

        pipeline.load_checkpoint(args.checkpoint)
        pipeline.load_index()

        logger.info(
            f"Type 1: Local VLM query | "
            f"model={config.vlm.model_name} | "
            f"mode={args.vlm_mode}"
        )

        result = pipeline.query_vlm(
            args.query_image,
            top_k=args.top_k,
            mode=args.vlm_mode,
            text_query=args.text_query,
        )

        output_path = os.path.join(
            config.output_dir, "vlm_report_result.json"
        )
        _save_result(result, output_path)
        _print_report_result(result)

    # ============================================================
    # Mode: query_ollama (Type 2: Ollama llava-llama3:8b)
    # ============================================================
    elif args.mode == "query_ollama":
        if not args.query_image:
            logger.error("--query_image is required.")
            sys.exit(1)

        # ── FORCE ollama backend when using query_ollama mode ──
        config.vlm_backend = "ollama"

        # Pre-flight check
        logger.info("Pre-flight: checking Ollama connection...")
        status = pipeline.check_ollama_connection()

        if not status["reachable"]:
            _print_ollama_status(status)
            logger.error("Ollama server not reachable. Aborting.")
            sys.exit(1)

        if not status["model_ready"]:
            logger.warning(
                f"Model '{config.ollama_vlm.model_name}' "
                f"not found. Attempting auto-pull..."
            )

        pipeline.load_checkpoint(args.checkpoint)
        pipeline.load_index()

        logger.info(
            f"Type 2: Ollama query | "
            f"{config.ollama_vlm.host} | "
            f"model={config.ollama_vlm.model_name} | "
            f"mode={args.vlm_mode}"
        )

        result = pipeline.query_ollama(
            args.query_image,
            top_k=args.top_k,
            mode=args.vlm_mode,
            text_query=args.text_query,
        )

        output_path = os.path.join(
            config.output_dir, "ollama_report_result.json"
        )
        _save_result(result, output_path)
        _print_report_result(result)

    # ============================================================
    # Mode: query_template (Type 3: Template with weighted probs)
    # ============================================================
    elif args.mode == "query_template":
        if not args.query_image:
            logger.error("--query_image is required.")
            sys.exit(1)

        pipeline.load_checkpoint(args.checkpoint)
        pipeline.load_index()

        logger.info("Type 3: Template report with weighted probabilities")

        result = pipeline.query_template(
            args.query_image,
            top_k=min(args.top_k, 3),  # Template uses top 1-3
            text_query=args.text_query,
        )

        output_path = os.path.join(
            config.output_dir, "template_report_result.json"
        )
        _save_result(result, output_path)
        print("\n" + result.get("report", "No report generated."))

    # ============================================================
    # Mode: query_visual (Type 4: Side-by-side gallery)
    # ============================================================
    elif args.mode == "query_visual":
        if not args.query_image:
            logger.error("--query_image is required.")
            sys.exit(1)

        pipeline.load_checkpoint(args.checkpoint)
        pipeline.load_index()

        logger.info("Type 4: Visual gallery (top 3-5 side-by-side)")

        result = pipeline.query_visual(
            args.query_image,
            top_k=args.top_k,
            text_query=args.text_query,
        )

        output_path = os.path.join(
            config.output_dir, "visual_report_result.json"
        )
        _save_result(result, output_path)
        print("\n" + result.get("report", "No report generated."))
        print(
            f"\n  Gallery images saved to: {config.output_dir}/"
        )

    # ============================================================
    # Mode: query_all (all 4 types at once)
    # ============================================================
    elif args.mode == "query_all":
        if not args.query_image:
            logger.error("--query_image is required.")
            sys.exit(1)

        pipeline.load_checkpoint(args.checkpoint)
        pipeline.load_index()

        logger.info("Generating all 4 output types...")

        all_outputs = pipeline.query_all_types(
            args.query_image,
            top_k=args.top_k,
            text_query=args.text_query,
        )

        # Save all results
        for type_name, output in all_outputs.items():
            output_path = os.path.join(
                config.output_dir,
                f"{type_name}_report_result.json",
            )
            _save_result(output, output_path)

        # Print summary
        _print_all_types_summary(all_outputs)

        # ============================================================
    # Mode: download_model — download MedGemma to local cache
    # ============================================================
    elif args.mode == "download_model":
        from vlm_report_generator import download_medgemma

        logger.info("=" * 60)
        logger.info("Downloading MedGemma model to local cache")
        logger.info("=" * 60)

        try:
            path = download_medgemma(
                model_name=config.vlm.model_name,
                cache_dir=config.vlm.local_model_dir,
                hf_token=config.vlm.hf_token,
            )
            logger.info(f"✅ Model downloaded to: {path}")
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            sys.exit(1)

    # ============================================================
    # Mode: check_model — check if MedGemma is downloaded
    # ============================================================
    elif args.mode == "check_model":
        from vlm_report_generator import check_medgemma_status

        status = check_medgemma_status(
            model_name=config.vlm.model_name,
            cache_dir=config.vlm.local_model_dir,
        )

        print("\n" + "=" * 55)
        print("  MEDGEMMA MODEL STATUS")
        print("=" * 55)

        for k, v in status.items():
            print(f"  {k}: {v}")

        if not status["is_cached"]:
            print(
                f"\n  Model not found. Download with:\n"
                f"    python main_pipeline.py --mode download_model"
            )

        print("=" * 55)
    
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()