# vlm_report_generator.py
"""
VLM-based medical report generation — MedGemma 4B-IT (local)
and remote Ollama (llava-llama3:8b) backends.

Local mode (default):
  Uses google/medgemma-4b-it — a 4B parameter medical VLM from
  Google, based on Gemma 2 with a SigLIP vision encoder.  It is
  instruction-tuned for medical visual question answering and
  report generation.  Downloaded to ./cache/huggingface.

  MedGemma natively supports multi-image inputs via its chat
  template.  For few-shot, we send the query image AND up to 3
  retrieved similar images together with their captions in a
  single conversation turn.

Ollama mode:
  Sends requests to an Ollama server running llava-llama3:8b.
  Falls back to text-grounded few-shot (captions only, single
  query image).

Requirements for local MedGemma:
  pip install transformers>=4.52.0 accelerate bitsandbytes pillow
  pip install huggingface_hub
  A HuggingFace token with access to google/medgemma-4b-it
  (accept the license at https://huggingface.co/google/medgemma-4b-it)
"""
import os
import io
import sys
import glob
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple
import logging
import json

from config import PipelineConfig, VLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================== #
#  Lazy imports
# ================================================================== #

HAS_MEDGEMMA = False

try:
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )
    HAS_MEDGEMMA = True
    logger.info("MedGemma / transformers imports successful")
except ImportError as e:
    logger.warning(
        f"transformers not fully available: {e}. "
        f"Install: pip install transformers>=4.52.0 accelerate"
    )

HAS_HF_HUB = False
try:
    from huggingface_hub import (
        snapshot_download,
        HfFolder,
        model_info,
    )
    HAS_HF_HUB = True
except ImportError:
    logger.warning(
        "huggingface_hub not installed. "
        "Install: pip install huggingface_hub"
    )

HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    pass


# ================================================================== #
#  MedGemma Model Manager (singleton with lazy loading)
# ================================================================== #

class MedGemmaModelManager:
    """
    Manages the local MedGemma 4B-IT model.
    Checks ./cache/huggingface for existing download;
    if not found, downloads automatically from HuggingFace.
    """

    _instance = None
    _model = None
    _processor = None
    _model_name = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ---------------------------------------------------------- #
    #  Token resolution
    # ---------------------------------------------------------- #

    @staticmethod
    def resolve_hf_token(config: VLMConfig) -> Optional[str]:
        """
        Resolve HuggingFace token from (in priority order):
          1. config.hf_token
          2. HF_TOKEN env var
          3. HUGGING_FACE_HUB_TOKEN env var
          4. huggingface-cli login cache
        """
        # 1. Explicit config
        if config.hf_token:
            return config.hf_token

        # 2. Environment variables
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        if token:
            return token

        # 3. huggingface-cli login cache
        if HAS_HF_HUB:
            try:
                token = HfFolder.get_token()
                if token:
                    return token
            except Exception:
                pass

        return None

    # ---------------------------------------------------------- #
    #  Check if model is already downloaded
    # ---------------------------------------------------------- #

    @staticmethod
    def _get_cached_model_path(
        model_name: str, cache_dir: str
    ) -> Optional[str]:
        """
        Check if the model already exists in the local cache.

        HuggingFace stores models in:
          cache_dir/models--{org}--{model}/snapshots/{hash}/

        Returns the snapshot path if found, else None.
        """
        # Convert model name to HF cache folder format
        # e.g. "google/medgemma-4b-it" -> "models--google--medgemma-4b-it"
        safe_name = model_name.replace("/", "--")
        model_cache_dir = os.path.join(
            cache_dir, f"models--{safe_name}"
        )

        if not os.path.isdir(model_cache_dir):
            return None

        # Look for snapshot directories
        snapshots_dir = os.path.join(model_cache_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return None

        # Find the latest snapshot
        snapshots = [
            d for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]

        if not snapshots:
            return None

        # Check that the snapshot has actual model files
        for snapshot in sorted(snapshots, reverse=True):
            snapshot_path = os.path.join(snapshots_dir, snapshot)
            # Look for key model files
            has_config = os.path.exists(
                os.path.join(snapshot_path, "config.json")
            )
            has_weights = (
                glob.glob(
                    os.path.join(snapshot_path, "*.safetensors")
                )
                or glob.glob(
                    os.path.join(snapshot_path, "*.bin")
                )
                or glob.glob(
                    os.path.join(snapshot_path, "model*.safetensors")
                )
            )

            if has_config and has_weights:
                return snapshot_path

            # Sometimes files are symlinks to blobs dir
            if has_config:
                return snapshot_path

        return None

    def is_model_cached(self, config: VLMConfig) -> bool:
        """Check if the model is already downloaded locally."""
        path = self._get_cached_model_path(
            config.model_name, config.local_model_dir
        )
        if path:
            logger.info(
                f"✅ Model found in cache: {path}"
            )
            return True
        else:
            logger.info(
                f"❌ Model NOT found in cache: "
                f"{config.local_model_dir}"
            )
            return False

    # ---------------------------------------------------------- #
    #  Download model
    # ---------------------------------------------------------- #

    def download_model(
        self,
        config: VLMConfig,
        force: bool = False,
    ) -> str:
        """
        Download MedGemma model to the local cache directory.

        Args:
            config: VLM configuration
            force:  If True, re-download even if already cached

        Returns:
            Path to the downloaded model directory
        """
        if not HAS_HF_HUB:
            raise RuntimeError(
                "huggingface_hub is required for downloading.\n"
                "Install: pip install huggingface_hub"
            )

        hf_token = self.resolve_hf_token(config)
        if hf_token is None:
            raise RuntimeError(
                "MedGemma is a gated model. You need a "
                "HuggingFace token with access.\n\n"
                "Steps:\n"
                "  1. Go to https://huggingface.co/google/"
                "medgemma-4b-it\n"
                "  2. Accept the license agreement\n"
                "  3. Set your token via ONE of:\n"
                "     a) config.py: hf_token = 'hf_...'\n"
                "     b) export HF_TOKEN=hf_...\n"
                "     c) huggingface-cli login\n"
                "     d) --hf_token hf_... (CLI)\n"
            )

        os.makedirs(config.local_model_dir, exist_ok=True)

        # Check if already downloaded
        if not force:
            cached_path = self._get_cached_model_path(
                config.model_name, config.local_model_dir
            )
            if cached_path:
                logger.info(
                    f"✅ Model already downloaded at: "
                    f"{cached_path}"
                )
                logger.info(
                    "Use force=True to re-download."
                )
                return cached_path

        # ── Verify access before downloading ──
        logger.info(
            f"Verifying access to {config.model_name}..."
        )
        try:
            info = model_info(
                config.model_name, token=hf_token
            )
            model_size = sum(
                s.size for s in (info.siblings or [])
                if s.size
            )
            if model_size > 0:
                size_gb = model_size / (1024 ** 3)
                logger.info(
                    f"Model size: ~{size_gb:.1f} GB"
                )
            logger.info(f"Access verified ✅")
        except Exception as e:
            raise RuntimeError(
                f"Cannot access {config.model_name}: {e}\n\n"
                f"Make sure you:\n"
                f"  1. Accepted the license at "
                f"https://huggingface.co/google/medgemma-4b-it\n"
                f"  2. Your token has 'read' permission\n"
                f"  3. You waited a few minutes after accepting"
            ) from e

        # ── Download ──
        logger.info("=" * 60)
        logger.info(
            f"Downloading {config.model_name} ..."
        )
        logger.info(
            f"Destination: {config.local_model_dir}"
        )
        logger.info(
            "This may take 10-30 minutes on first run."
        )
        logger.info("=" * 60)

        try:
            downloaded_path = snapshot_download(
                repo_id=config.model_name,
                cache_dir=config.local_model_dir,
                token=hf_token,
                # Download everything needed
                ignore_patterns=[
                    "*.md",
                    "*.txt",
                    ".gitattributes",
                ],
            )

            logger.info("=" * 60)
            logger.info(f"✅ Download complete!")
            logger.info(f"   Path: {downloaded_path}")
            logger.info("=" * 60)

            return downloaded_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise RuntimeError(
                f"Failed to download {config.model_name}: {e}\n\n"
                f"Common fixes:\n"
                f"  - Check internet connection\n"
                f"  - Verify HF token and license acceptance\n"
                f"  - Ensure enough disk space (~8 GB)\n"
                f"  - Try: huggingface-cli download "
                f"{config.model_name} "
                f"--cache-dir {config.local_model_dir}"
            ) from e

    # ---------------------------------------------------------- #
    #  Load model (download if needed)
    # ---------------------------------------------------------- #

    def load_model(self, config: VLMConfig):
        """
        Load MedGemma 4B-IT model and processor.
        If not found in cache, downloads automatically first.
        """
        if not HAS_MEDGEMMA:
            raise RuntimeError(
                "MedGemma dependencies not installed.\n"
                "Install: pip install transformers>=4.52.0 "
                "accelerate bitsandbytes\n"
                "Or use --vlm_backend ollama."
            )

        # Return cached model if already in memory
        if (
            config.cache_model
            and self._model is not None
            and self._model_name == config.model_name
        ):
            logger.info(
                f"MedGemma already in memory: "
                f"{config.model_name}"
            )
            return self._model, self._processor

        # ── Step 1: Check local cache, download if missing ──
        logger.info(
            f"Checking for {config.model_name} in "
            f"{config.local_model_dir} ..."
        )

        if not self.is_model_cached(config):
            logger.info(
                f"Model not found locally. "
                f"Starting download..."
            )
            try:
                self.download_model(config)
            except Exception as e:
                raise RuntimeError(
                    f"Could not download MedGemma: {e}\n"
                    f"You can also download manually:\n"
                    f"  huggingface-cli download "
                    f"{config.model_name} "
                    f"--cache-dir {config.local_model_dir}\n"
                    f"Or use --vlm_backend ollama."
                ) from e
        else:
            logger.info("Model found in local cache ✅")

        # ── Step 2: Resolve token ──
        hf_token = self.resolve_hf_token(config)
        if hf_token is None:
            raise RuntimeError(
                "HuggingFace token required for MedGemma.\n"
                "Set it in config.py, env var, or CLI."
            )

        os.makedirs(config.local_model_dir, exist_ok=True)

        # ── Step 3: Torch dtype ──
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(
            config.torch_dtype, torch.bfloat16
        )

        # ── Step 4: Quantization config ──
        quantization_config = None
        if config.load_in_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("Using 4-bit quantization")
            except Exception as e:
                logger.warning(
                    f"4-bit quantization setup failed: {e}"
                )
        elif config.load_in_8bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                logger.info("Using 8-bit quantization")
            except Exception as e:
                logger.warning(
                    f"8-bit quantization setup failed: {e}"
                )

        # ── Step 5: Load processor ──
        logger.info("Loading MedGemma processor...")
        self._processor = AutoProcessor.from_pretrained(
            config.model_name,
            token=hf_token,
            cache_dir=config.local_model_dir,
            trust_remote_code=True,
        )

        # ── Step 6: Load model weights ──
        logger.info(
            "Loading MedGemma model weights "
            "(this may take a minute)..."
        )
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": config.device_map,
            "token": hf_token,
            "cache_dir": config.local_model_dir,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = (
                quantization_config
            )

        self._model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            **model_kwargs,
        )
        self._model.eval()
        self._model_name = config.model_name

        total_params = sum(
            p.numel() for p in self._model.parameters()
        )
        logger.info("=" * 60)
        logger.info(
            f"✅ MedGemma loaded successfully!"
        )
        logger.info(
            f"   Model:  {config.model_name}"
        )
        logger.info(
            f"   Params: {total_params / 1e9:.1f}B"
        )
        logger.info(
            f"   Dtype:  {config.torch_dtype}"
        )
        logger.info(
            f"   Cache:  {config.local_model_dir}"
        )
        if quantization_config:
            quant = (
                "4-bit" if config.load_in_4bit else "8-bit"
            )
            logger.info(f"   Quant:  {quant}")
        logger.info("=" * 60)

        return self._model, self._processor

    # ---------------------------------------------------------- #
    #  Unload
    # ---------------------------------------------------------- #

    def unload_model(self):
        """Free model memory."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._model_name = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("MedGemma model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# ================================================================== #
#  Standalone download function
# ================================================================== #

def download_medgemma(
    model_name: str = "google/medgemma-4b-it",
    cache_dir: str = "./cache/huggingface",
    hf_token: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Standalone function to download MedGemma to a local folder.

    Can be called directly:
        from vlm_report_generator import download_medgemma
        download_medgemma(hf_token="hf_...")

    Or from CLI:
        python vlm_report_generator.py --download

    Returns:
        Path to the downloaded model
    """
    config = VLMConfig(
        model_name=model_name,
        local_model_dir=cache_dir,
        hf_token=hf_token,
    )

    manager = MedGemmaModelManager.get_instance()
    return manager.download_model(config, force=force)


def check_medgemma_status(
    model_name: str = "google/medgemma-4b-it",
    cache_dir: str = "./cache/huggingface",
) -> Dict:
    """
    Check download status of MedGemma.

    Returns:
        Dict with status information
    """
    config = VLMConfig(
        model_name=model_name,
        local_model_dir=cache_dir,
    )

    manager = MedGemmaModelManager.get_instance()

    cached_path = manager._get_cached_model_path(
        config.model_name, config.local_model_dir
    )

    status = {
        "model_name": model_name,
        "cache_dir": cache_dir,
        "is_cached": cached_path is not None,
        "cached_path": cached_path,
        "transformers_available": HAS_MEDGEMMA,
        "huggingface_hub_available": HAS_HF_HUB,
    }

    if cached_path:
        # Calculate size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(
            cached_path
        ):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)

        status["cache_size_gb"] = round(
            total_size / (1024 ** 3), 2
        )

        # List key files
        key_files = []
        for f in os.listdir(cached_path):
            if f.endswith(
                (".json", ".safetensors", ".bin", ".model")
            ):
                key_files.append(f)
        status["key_files"] = sorted(key_files)

    # Check token
    token = MedGemmaModelManager.resolve_hf_token(config)
    status["has_token"] = token is not None

    return status


# ================================================================== #
#  MedGemma Prompt Builder
# ================================================================== #

class MedGemmaPromptBuilder:
    """
    Builds chat-style prompts for MedGemma 4B-IT.

    MedGemma uses the Gemma chat template and supports multiple
    images in a single turn via <image> placeholder tokens or
    by passing images in the processor.

    For few-shot we embed the query image AND up to 3 retrieved
    images directly into the conversation so the model can
    visually compare them.
    """

    @staticmethod
    def build_few_shot_messages(
        num_query_images: int = 1,
        example_captions: Optional[List[str]] = None,
        example_scores: Optional[List[float]] = None,
        num_example_images: int = 0,
        modality_context: str = "chest X-ray radiology",
        include_scores: bool = True,
        detected_conditions: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Build a chat message list for MedGemma.

        Returns:
            list of message dicts compatible with
            processor.apply_chat_template
        """
        # ── System-level context ──
        system_text = (
            f"You are an expert radiologist specializing in "
            f"{modality_context}. Generate accurate, detailed, "
            f"and clinically relevant radiology reports. Use "
            f"standard radiology terminology. Do NOT hallucinate "
            f"findings that are not visible in the images."
        )

        # ── Condition context ──
        condition_text = ""
        if detected_conditions:
            pathological = {
                k: v for k, v in detected_conditions.items()
                if k not in ("normal", "support_devices")
            }
            if pathological:
                cond_strs = []
                for k, v in sorted(
                    pathological.items(),
                    key=lambda x: x[1].get("avg_score", 0),
                    reverse=True,
                ):
                    freq = v.get(
                        "frequency",
                        v.get("weighted_probability", 0),
                    )
                    cond_strs.append(
                        f"{k.replace('_', ' ')} "
                        f"({freq * 100:.0f}%)"
                    )
                condition_text = (
                    "\n\nDatabase analysis of visually similar "
                    "images suggests these conditions may be "
                    "present: " + ", ".join(cond_strs) + ". "
                    "Use this as additional context but rely "
                    "primarily on what you see in the images."
                )

        # ── Build user content blocks ──
        user_content = []

        user_content.append({
            "type": "text",
            "text": system_text + condition_text,
        })

        # ── Example images and captions ──
        if example_captions:
            user_content.append({
                "type": "text",
                "text": (
                    "\n\nBelow are the most similar cases from "
                    "the medical database. Each has a reference "
                    "image and its radiology report:"
                ),
            })

            for i, caption in enumerate(example_captions):
                score_str = ""
                if (
                    include_scores
                    and example_scores
                    and i < len(example_scores)
                ):
                    score_str = (
                        f" (similarity score: "
                        f"{example_scores[i]:.3f})"
                    )

                if i < num_example_images:
                    user_content.append({
                        "type": "image",
                    })
                    user_content.append({
                        "type": "text",
                        "text": (
                            f"\nSimilar Case {i + 1}"
                            f"{score_str}:\n{caption}"
                        ),
                    })
                else:
                    user_content.append({
                        "type": "text",
                        "text": (
                            f"\nSimilar Case {i + 1}"
                            f"{score_str} (text only):\n"
                            f"{caption}"
                        ),
                    })

        # ── Query image ──
        user_content.append({
            "type": "text",
            "text": (
                "\n\nNow analyze the following query "
                f"{modality_context} image and generate a "
                "comprehensive radiology report."
            ),
        })
        user_content.append({
            "type": "image",
        })
        user_content.append({
            "type": "text",
            "text": (
                "\nGenerate a structured report with:\n"
                "FINDINGS: (describe all observations)\n"
                "IMPRESSION: (summarize key diagnoses and "
                "recommendations)"
            ),
        })

        messages = [
            {"role": "user", "content": user_content},
        ]

        return messages

    @staticmethod
    def build_zero_shot_messages(
        modality_context: str = "chest X-ray radiology",
        detected_conditions: Optional[Dict] = None,
    ) -> List[Dict]:
        """Build zero-shot messages (query image only)."""

        system_text = (
            f"You are an expert radiologist specializing in "
            f"{modality_context}. Analyze the following medical "
            f"image and generate a detailed radiology report."
        )

        condition_text = ""
        if detected_conditions:
            pathological = {
                k: v for k, v in detected_conditions.items()
                if k not in ("normal", "support_devices")
            }
            if pathological:
                cond_strs = [
                    k.replace("_", " ")
                    for k in sorted(
                        pathological.keys(),
                        key=lambda x: pathological[x].get(
                            "avg_score",
                            pathological[x].get(
                                "weighted_probability", 0
                            ),
                        ),
                        reverse=True,
                    )
                ]
                condition_text = (
                    "\nAutomated analysis suggests possible: "
                    + ", ".join(cond_strs) + ". "
                    "Verify these findings in the image."
                )

        user_content = [
            {"type": "text", "text": system_text + condition_text},
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "\nGenerate a structured report with:\n"
                    "FINDINGS: (detailed observations)\n"
                    "IMPRESSION: (summary of key diagnoses)"
                ),
            },
        ]

        messages = [
            {"role": "user", "content": user_content},
        ]

        return messages


# ================================================================== #
#  VLM Report Generator (unified interface)
# ================================================================== #

class VLMReportGenerator:
    """
    Generates medical reports using a Vision-Language Model.

    Supports two backends:
      1. Local: MedGemma 4B-IT (google/medgemma-4b-it)
         — Multi-image support (query + up to 3 retrieved images)
         — Auto-downloads to ./cache/huggingface if not found
      2. Ollama: llava-llama3:8b on a remote Ollama server
    """

    def __init__(self, config: PipelineConfig, dataset=None):
        self.config = config
        self.vlm_config = config.vlm
        self.dataset = dataset

        # Local MedGemma (lazy loaded)
        self.model_manager = MedGemmaModelManager.get_instance()
        self.prompt_builder = MedGemmaPromptBuilder()

        # Ollama generator (lazy loaded)
        self._ollama_generator = None

    def set_dataset(self, dataset):
        """Set the reference dataset for loading retrieved images."""
        self.dataset = dataset

    # ---------------------------------------------------------- #
    #  Ollama generator (lazy init)
    # ---------------------------------------------------------- #

    def _get_ollama_generator(self):
        """Get or create the Ollama VLM report generator."""
        if self._ollama_generator is None:
            from ollama_vlm_report_generator import (
                OllamaVLMReportGenerator,
            )
            self._ollama_generator = OllamaVLMReportGenerator(
                self.config
            )
        return self._ollama_generator

    # ---------------------------------------------------------- #
    #  Backend detection
    # ---------------------------------------------------------- #

    @property
    def is_available(self) -> bool:
        backend = self.config.vlm_backend

        if backend == "ollama":
            try:
                gen = self._get_ollama_generator()
                return gen.is_available
            except Exception as e:
                logger.warning(
                    f"Ollama backend not available: {e}"
                )
                return False

        elif backend == "local":
            return HAS_MEDGEMMA and self.vlm_config.enabled

        return False

    @property
    def active_backend(self) -> str:
        return self.config.vlm_backend

    # ---------------------------------------------------------- #
    #  Load retrieved images from dataset
    # ---------------------------------------------------------- #

    def _load_image_by_index(
        self, original_index: int
    ) -> Optional[Image.Image]:
        """Load an image from the dataset by its original index."""
        if self.dataset is None:
            return None

        try:
            if hasattr(self.dataset, "dataset"):
                item = self.dataset.dataset[original_index]
            else:
                item = self.dataset[original_index]

            raw = item.get("image", None)
            if raw is None:
                return None

            if isinstance(raw, Image.Image):
                return raw.convert("RGB")
            if isinstance(raw, str):
                return Image.open(raw).convert("RGB")
            if isinstance(raw, dict):
                if "bytes" in raw and raw["bytes"]:
                    return Image.open(
                        io.BytesIO(raw["bytes"])
                    ).convert("RGB")
                if "path" in raw and raw["path"]:
                    return Image.open(
                        raw["path"]
                    ).convert("RGB")

            arr = np.array(raw)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return Image.fromarray(
                arr.astype(np.uint8)
            ).convert("RGB")

        except Exception as e:
            logger.warning(
                f"Could not load image at index "
                f"{original_index}: {e}"
            )
            return None

    def _collect_example_images(
        self,
        retrieval_results: List[Dict],
        max_images: int = 3,
    ) -> List[Optional[Image.Image]]:
        """
        Try to load images for the top retrieved results.
        """
        images = []
        for result in retrieval_results[:max_images]:
            orig_idx = result.get("original_index")
            if orig_idx is not None:
                img = self._load_image_by_index(orig_idx)
                images.append(img)
            else:
                images.append(None)
        return images

    # ---------------------------------------------------------- #
    #  Local MedGemma generation
    # ---------------------------------------------------------- #

    def _generate_local(
        self,
        messages: List[Dict],
        images: List[Image.Image],
    ) -> str:
        """
        Run local MedGemma generation.
        Auto-downloads model if not found in cache.
        """
        model, processor = self.model_manager.load_model(
            self.vlm_config
        )

        input_text = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = processor(
            text=input_text,
            images=images if images else None,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.vlm_config.max_new_tokens,
                temperature=self.vlm_config.temperature,
                top_p=self.vlm_config.top_p,
                repetition_penalty=(
                    self.vlm_config.repetition_penalty
                ),
                do_sample=self.vlm_config.do_sample,
                num_beams=self.vlm_config.num_beams,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]

        result = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return result.strip()

    # ---------------------------------------------------------- #
    #  Public API — few-shot
    # ---------------------------------------------------------- #

    def generate_report_few_shot(
        self,
        query_image: Image.Image,
        retrieval_results: List[Dict],
        detected_conditions: Optional[Dict] = None,
        num_examples: Optional[int] = None,
    ) -> Dict:
        """
        Few-shot report generation.
        MedGemma: query image + 1-3 retrieved images + captions.
        Ollama: query image + captions only (text-grounded).
        """
        backend = self.config.vlm_backend

        if backend == "ollama":
            gen = self._get_ollama_generator()
            return gen.generate_report_few_shot(
                query_image=query_image,
                retrieval_results=retrieval_results,
                detected_conditions=detected_conditions,
                num_examples=num_examples,
            )

        if not HAS_MEDGEMMA or not self.vlm_config.enabled:
            return {
                "report": (
                    "MedGemma not available. Install:\n"
                    "  pip install transformers>=4.52.0 "
                    "accelerate\n"
                    "Or use --vlm_backend ollama."
                ),
                "method": "vlm_few_shot",
                "success": False,
                "error": "MedGemma dependencies not installed",
            }

        if num_examples is None:
            num_examples = self.vlm_config.num_examples

        num_examples = max(1, min(3, num_examples))
        examples = retrieval_results[:num_examples]

        try:
            example_captions = [
                r.get("caption", "No caption available")
                for r in examples
            ]
            example_scores = [
                r.get("score", 0.0) for r in examples
            ]

            example_images_raw = self._collect_example_images(
                examples, max_images=num_examples
            )

            example_images = []
            for img in example_images_raw:
                if img is not None:
                    example_images.append(img)

            num_example_images = len(example_images)

            logger.info(
                f"MedGemma few-shot: {len(example_captions)} "
                f"text examples, {num_example_images} images "
                f"loaded, query image provided"
            )

            messages = self.prompt_builder.build_few_shot_messages(
                num_query_images=1,
                example_captions=example_captions,
                example_scores=example_scores,
                num_example_images=num_example_images,
                modality_context=(
                    self.vlm_config.modality_context
                ),
                include_scores=(
                    self.vlm_config.include_scores_in_prompt
                ),
                detected_conditions=(
                    detected_conditions
                    if self.vlm_config.include_conditions_context
                    else None
                ),
            )

            all_images = example_images + [query_image]

            report_text = self._generate_local(
                messages, all_images
            )

            formatted = self._format_report(
                report_text,
                retrieval_results=examples,
                mode="few_shot",
                backend="local",
                num_images=num_example_images + 1,
            )

            return {
                "report": formatted,
                "raw_vlm_output": report_text,
                "method": "vlm_few_shot",
                "num_examples": len(example_captions),
                "num_images_sent": num_example_images + 1,
                "success": True,
                "model": self.vlm_config.model_name,
                "backend": "local",
            }

        except Exception as e:
            logger.error(
                f"MedGemma few-shot failed: {e}",
                exc_info=True,
            )
            return {
                "report": f"MedGemma generation failed: {e}",
                "method": "vlm_few_shot",
                "success": False,
                "error": str(e),
            }

    # ---------------------------------------------------------- #
    #  Public API — zero-shot
    # ---------------------------------------------------------- #

    def generate_report_zero_shot(
        self,
        query_image: Image.Image,
        detected_conditions: Optional[Dict] = None,
    ) -> Dict:
        """Zero-shot report (query image only)."""
        backend = self.config.vlm_backend

        if backend == "ollama":
            gen = self._get_ollama_generator()
            return gen.generate_report_zero_shot(
                query_image=query_image,
                detected_conditions=detected_conditions,
            )

        if not HAS_MEDGEMMA or not self.vlm_config.enabled:
            return {
                "report": (
                    "MedGemma not available. "
                    "Use --vlm_backend ollama."
                ),
                "method": "vlm_zero_shot",
                "success": False,
                "error": "Dependencies not installed",
            }

        try:
            logger.info("MedGemma zero-shot generation...")

            messages = (
                self.prompt_builder.build_zero_shot_messages(
                    modality_context=(
                        self.vlm_config.modality_context
                    ),
                    detected_conditions=(
                        detected_conditions
                        if self.vlm_config
                        .include_conditions_context
                        else None
                    ),
                )
            )

            report_text = self._generate_local(
                messages, [query_image]
            )

            formatted = self._format_report(
                report_text,
                mode="zero_shot",
                backend="local",
                num_images=1,
            )

            return {
                "report": formatted,
                "raw_vlm_output": report_text,
                "method": "vlm_zero_shot",
                "num_examples": 0,
                "num_images_sent": 1,
                "success": True,
                "model": self.vlm_config.model_name,
                "backend": "local",
            }

        except Exception as e:
            logger.error(
                f"MedGemma zero-shot failed: {e}",
                exc_info=True,
            )
            return {
                "report": f"MedGemma generation failed: {e}",
                "method": "vlm_zero_shot",
                "success": False,
                "error": str(e),
            }

    # ---------------------------------------------------------- #
    #  Ollama-specific public API (force Ollama)
    # ---------------------------------------------------------- #

    def generate_report_ollama_few_shot(
        self,
        query_image: Image.Image,
        retrieval_results: List[Dict],
        detected_conditions: Optional[Dict] = None,
        num_examples: Optional[int] = None,
    ) -> Dict:
        """Force Ollama backend for few-shot generation."""
        gen = self._get_ollama_generator()
        return gen.generate_report_few_shot(
            query_image=query_image,
            retrieval_results=retrieval_results,
            detected_conditions=detected_conditions,
            num_examples=num_examples,
        )

    def generate_report_ollama_zero_shot(
        self,
        query_image: Image.Image,
        detected_conditions: Optional[Dict] = None,
    ) -> Dict:
        """Force Ollama backend for zero-shot generation."""
        gen = self._get_ollama_generator()
        return gen.generate_report_zero_shot(
            query_image=query_image,
            detected_conditions=detected_conditions,
        )

    # ---------------------------------------------------------- #
    #  Report formatting
    # ---------------------------------------------------------- #

    def _format_report(
        self,
        raw_text: str,
        retrieval_results: Optional[List[Dict]] = None,
        mode: str = "few_shot",
        backend: str = "local",
        num_images: int = 1,
    ) -> str:
        """Format raw VLM output into a structured report."""
        lines = []
        lines.append("=" * 65)

        if backend == "ollama":
            model_label = (
                f"llava-llama3:8b via Ollama "
                f"@ {self.config.ollama_vlm.host}"
            )
        else:
            model_label = (
                f"MedGemma 4B-IT ({self.vlm_config.model_name})"
            )

        mode_label = (
            "Multi-Image Few-Shot" if mode == "few_shot"
            else "Zero-Shot"
        )

        lines.append(f"  AI-GENERATED RADIOLOGY REPORT")
        lines.append(f"  Model: {model_label}")
        lines.append(
            f"  Mode:  {mode_label} | Backend: {backend}"
        )
        lines.append("=" * 65)
        lines.append("")

        cleaned = self._clean_vlm_output(raw_text)
        lines.append(cleaned)

        lines.append("")
        lines.append("-" * 65)

        if mode == "few_shot" and retrieval_results:
            lines.append(
                f"  Generated using {len(retrieval_results)} "
                f"similar cases as context "
                f"({num_images} images sent to model)"
            )
            scores = [
                r.get("score", 0) for r in retrieval_results
            ]
            if scores:
                lines.append(
                    f"  Similarity score range: "
                    f"{min(scores):.3f} – {max(scores):.3f}"
                )
        else:
            lines.append(
                f"  Generated without example context "
                f"({num_images} image(s))"
            )

        lines.append(
            f"  Temperature: {self.vlm_config.temperature} | "
            f"Max tokens: {self.vlm_config.max_new_tokens}"
        )
        lines.append("=" * 65)

        return "\n".join(lines)

    @staticmethod
    def _clean_vlm_output(text: str) -> str:
        """Clean up common artifacts in VLM output."""
        artifacts = [
            "<end_of_turn>", "<start_of_turn>",
            "<eos>", "</s>", "<s>",
            "<bos>", "<pad>",
            "model\n",
            "Report:", "report:",
        ]
        for artifact in artifacts:
            text = text.replace(artifact, "")

        text = text.strip()

        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith(
                "**FINDINGS"
            ) or upper.startswith("FINDINGS"):
                content = ""
                if ":" in line:
                    content = line.split(":", 1)[-1]
                line = "FINDINGS:" + content
            elif upper.startswith(
                "**IMPRESSION"
            ) or upper.startswith("IMPRESSION"):
                content = ""
                if ":" in line:
                    content = line.split(":", 1)[-1]
                line = "IMPRESSION:" + content
            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines).strip()

        has_findings = "FINDINGS" in text.upper()
        has_impression = "IMPRESSION" in text.upper()

        if not has_findings and not has_impression:
            text = (
                f"FINDINGS:\n{text}\n\n"
                f"IMPRESSION:\nSee findings above."
            )

        return text

    # ---------------------------------------------------------- #
    #  Model management
    # ---------------------------------------------------------- #

    def unload_model(self):
        """Free local VLM model memory."""
        self.model_manager.unload_model()

    def reset_ollama_connection(self):
        """Reset Ollama connection state."""
        if self._ollama_generator is not None:
            self._ollama_generator.reset_connection()


# ================================================================== #
#  CLI — standalone download / status check
# ================================================================== #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MedGemma model management"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download MedGemma to local cache",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Check MedGemma download status",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--model", type=str,
        default="google/medgemma-4b-it",
        help="Model name",
    )
    parser.add_argument(
        "--cache_dir", type=str,
        default="./cache/huggingface",
        help="Cache directory",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token",
    )

    args = parser.parse_args()

    if args.status:
        print("\n" + "=" * 55)
        print("  MEDGEMMA STATUS CHECK")
        print("=" * 55)

        status = check_medgemma_status(
            model_name=args.model,
            cache_dir=args.cache_dir,
        )

        for k, v in status.items():
            print(f"  {k}: {v}")

        print("=" * 55)

    elif args.download:
        print("\n" + "=" * 55)
        print("  DOWNLOADING MEDGEMMA")
        print("=" * 55)

        try:
            path = download_medgemma(
                model_name=args.model,
                cache_dir=args.cache_dir,
                hf_token=args.hf_token,
                force=args.force,
            )
            print(f"\n✅ Model ready at: {path}")
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            sys.exit(1)

    else:
        parser.print_help()