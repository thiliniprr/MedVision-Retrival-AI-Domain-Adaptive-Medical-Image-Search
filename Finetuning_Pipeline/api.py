# api.py
"""
REST API for the Medical Image Retrieval & Report Generation Pipeline.

Supports:
  - Local MedGemma 4B-IT (google/medgemma-4b-it) — default
  - Ollama remote VLM (llava-llama3:8b) — optional
  - Template reports (no VLM needed)
  - Visual gallery generation

Endpoints:
  POST /api/retrieve         — Retrieve top-K similar images
  POST /api/generate-report  — Generate report (local or ollama)
  POST /api/index/append     — Append new data to FAISS index
  POST /api/feedback         — Submit feedback + update index

  GET  /api/health           — Health check
  GET  /api/status           — Pipeline status
  POST /api/load             — Load checkpoint + index

  GET  /api/vlm/status       — MedGemma download/load status
  POST /api/vlm/download     — Download MedGemma model
  POST /api/vlm/load         — Pre-load MedGemma into memory
  POST /api/vlm/unload       — Free MedGemma from memory

Usage:
  python api.py                                    # defaults
  python api.py --port 8080                        # custom port
  python api.py --vlm_backend local --vlm_4bit     # 4-bit quantized
  python api.py --vlm_backend ollama               # use Ollama instead
"""

import os
import io
import csv
import json
import uuid
import shutil
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image

from fastapi import (
    FastAPI, File, UploadFile, Form, HTTPException, Query,
    Request,
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import (
    PipelineConfig, VLMConfig, OllamaVLMConfig,
)

os.environ['HF_TOKEN'] = VLMConfig.hf_token
# 1. Find exactly where this Python file is sitting on the server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Attach '/cache/huggingface' to that exact location
os.environ['HF_HOME'] = os.path.join(BASE_DIR, 'cache', 'huggingface')

from main_pipeline import MedicalImageRetrievalPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)



# ================================================================== #
#  Query Log Manager
# ================================================================== #

class QueryLogManager:
    """Persistent query log stored as JSON Lines (.jsonl)."""

    def __init__(
        self, log_path: str = "./output/query_log.jsonl"
    ):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                pass
            logger.info(f"Created new query log: {log_path}")
        else:
            logger.info(f"Using existing query log: {log_path}")

    def add_entry(
        self,
        query_id: str,
        image_filename: str,
        generated_caption: str,
        feedback: str,
        report_method: Optional[str] = None,
        retrieval_scores: Optional[List[float]] = None,
        num_retrieved: int = 0,
        extra_metadata: Optional[Dict] = None,
    ) -> Dict:
        entry = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "image_filename": image_filename,
            "generated_caption": generated_caption,
            "user_feedback": feedback,
            "report_method": report_method,
            "num_retrieved": num_retrieved,
            "retrieval_scores": retrieval_scores or [],
        }

        if extra_metadata:
            entry["metadata"] = extra_metadata

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        logger.info(
            f"Query log entry added: {query_id} | "
            f"feedback length: {len(feedback)} chars"
        )
        return entry

    def get_all_entries(self) -> List[Dict]:
        entries = []
        if not os.path.exists(self.log_path):
            return entries

        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def get_entries_by_date(self, date_str: str) -> List[Dict]:
        return [
            e for e in self.get_all_entries()
            if e.get("date") == date_str
        ]

    def get_entry_count(self) -> int:
        if not os.path.exists(self.log_path):
            return 0
        count = 0
        with open(self.log_path, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


# ================================================================== #
#  CSV Data Loader (for index append)
# ================================================================== #

class CSVDataLoader:
    """Load image-caption pairs from a folder with a CSV file."""

    IMAGE_COLUMNS = [
        "image", "image_filename", "filename",
        "image_path", "file", "img", "path",
    ]
    CAPTION_COLUMNS = [
        "caption", "findings", "text", "report",
        "impression", "description", "label",
    ]

    @staticmethod
    def find_csv_file(folder_path: str) -> Optional[str]:
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".csv"):
                return os.path.join(folder_path, fname)
        return None

    @staticmethod
    def detect_columns(headers: List[str]) -> tuple:
        headers_lower = [h.strip().lower() for h in headers]

        image_col = None
        caption_col = None

        for pattern in CSVDataLoader.IMAGE_COLUMNS:
            for i, h in enumerate(headers_lower):
                if pattern in h:
                    image_col = i
                    break
            if image_col is not None:
                break

        for pattern in CSVDataLoader.CAPTION_COLUMNS:
            for i, h in enumerate(headers_lower):
                if pattern in h and i != image_col:
                    caption_col = i
                    break
            if caption_col is not None:
                break

        if image_col is None:
            image_col = 0
        if caption_col is None:
            caption_col = 1 if len(headers) > 1 else 0

        return image_col, caption_col

    @staticmethod
    def load_from_folder(folder_path: str) -> List[Dict]:
        csv_path = CSVDataLoader.find_csv_file(folder_path)
        if csv_path is None:
            raise FileNotFoundError(
                f"No .csv file found in {folder_path}"
            )

        logger.info(f"Loading data from CSV: {csv_path}")

        pairs = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, None)

            if headers is None:
                raise ValueError("CSV file is empty")

            image_col, caption_col = (
                CSVDataLoader.detect_columns(headers)
            )
            logger.info(
                f"Detected columns — "
                f"image: '{headers[image_col]}' "
                f"(col {image_col}), "
                f"caption: '{headers[caption_col]}' "
                f"(col {caption_col})"
            )

            for row_num, row in enumerate(reader, start=2):
                if len(row) <= max(image_col, caption_col):
                    continue

                image_ref = row[image_col].strip()
                caption = row[caption_col].strip()

                if not caption:
                    continue

                image_path = None
                if os.path.isabs(image_ref):
                    if os.path.exists(image_ref):
                        image_path = image_ref
                else:
                    candidate = os.path.join(
                        folder_path, image_ref
                    )
                    if os.path.exists(candidate):
                        image_path = candidate

                if image_path is None:
                    base_name = os.path.splitext(image_ref)[0]
                    for ext in [
                        ".png", ".jpg", ".jpeg",
                        ".dicom", ".dcm",
                    ]:
                        candidate = os.path.join(
                            folder_path, base_name + ext
                        )
                        if os.path.exists(candidate):
                            image_path = candidate
                            break

                if image_path is None:
                    continue

                pairs.append({
                    "image_path": image_path,
                    "caption": caption,
                })

        logger.info(
            f"Loaded {len(pairs)} image-caption pairs "
            f"from {csv_path}"
        )
        return pairs


# ================================================================== #
#  API Application Factory
# ================================================================== #

def create_app(
    config: Optional[PipelineConfig] = None,
    checkpoint_name: str = "final_model",
    auto_load: bool = True,
) -> FastAPI:
    """
    Create the FastAPI application with all endpoints.

    Supports:
      - Local MedGemma 4B-IT (default)
      - Ollama remote VLM (optional)
      - Template reports (no VLM)
    """

    if config is None:
        config = PipelineConfig()

    pipeline = MedicalImageRetrievalPipeline(config)
    query_log = QueryLogManager(
        os.path.join(config.output_dir, "query_log.jsonl")
    )

    # Track state
    state = {
        "pipeline_ready": False,
        "vlm_backend": config.vlm_backend,
        "vlm_model_cached": False,
        "vlm_model_loaded": False,
        "ollama_ready": False,
        "startup_time": None,
        "request_count": 0,
        "total_generation_time": 0.0,
    }

    # ── Lifespan (startup + shutdown) ──
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("=" * 60)
        logger.info("  STARTING MEDICAL IMAGE RETRIEVAL API")
        logger.info(f"  VLM backend: {config.vlm_backend}")
        logger.info("=" * 60)

        state["startup_time"] = datetime.now()

        # Step 1: Load CLIP checkpoint + FAISS index
        if auto_load:
            try:
                logger.info(
                    f"Loading checkpoint '{checkpoint_name}' "
                    f"and FAISS index..."
                )
                pipeline.load_checkpoint(checkpoint_name)
                pipeline.load_index()
                state["pipeline_ready"] = True
                logger.info("✅ Pipeline loaded and ready")
            except Exception as e:
                logger.warning(
                    f"⚠️  Auto-load failed: {e}. "
                    f"Pipeline not ready."
                )

        # ══════════════════════════════════════════════════════
        # Step 2: Auto-download + auto-load MedGemma on startup
        # ══════════════════════════════════════════════════════
        if config.vlm_backend == "local" and config.vlm.enabled:
            logger.info(
                f"Setting up local VLM: "
                f"{config.vlm.model_name}"
            )

            # 2a. Check if model is cached
            try:
                from vlm_report_generator import (
                    check_medgemma_status,
                    download_medgemma,
                    MedGemmaModelManager,
                )

                vlm_status = check_medgemma_status(
                    model_name=config.vlm.model_name,
                    cache_dir=config.vlm.local_model_dir,
                )
                state["vlm_model_cached"] = vlm_status.get(
                    "is_cached", False
                )

                # 2b. Download if not cached
                if not state["vlm_model_cached"]:
                    logger.info(
                        "⬇️  MedGemma not found in cache. "
                        "Downloading now..."
                    )
                    logger.info(
                        "   This may take 10-30 minutes on "
                        "first run."
                    )
                    try:
                        path = download_medgemma(
                            model_name=config.vlm.model_name,
                            cache_dir=config.vlm.local_model_dir,
                            hf_token=config.vlm.hf_token,
                        )
                        state["vlm_model_cached"] = True
                        logger.info(
                            f"✅ MedGemma downloaded to: {path}"
                        )
                    except Exception as e:
                        logger.error(
                            f"❌ MedGemma download failed: {e}"
                        )
                        logger.error(
                            "   VLM reports will not be available. "
                            "Template reports still work."
                        )
                else:
                    logger.info(
                        f"✅ MedGemma found in cache: "
                        f"{config.vlm.model_name}"
                    )

                # 2c. Pre-load model into memory
                if state["vlm_model_cached"]:
                    logger.info(
                        "Loading MedGemma into memory "
                        "(this may take 1-2 minutes)..."
                    )
                    try:
                        manager = (
                            MedGemmaModelManager.get_instance()
                        )
                        manager.load_model(config.vlm)
                        state["vlm_model_loaded"] = True
                        logger.info(
                            "✅ MedGemma loaded and ready"
                        )
                    except Exception as e:
                        logger.warning(
                            f"⚠️  MedGemma pre-load failed: {e}"
                        )
                        logger.info(
                            "   Model will auto-load on first "
                            "VLM query instead."
                        )

            except ImportError as e:
                logger.warning(
                    f"⚠️  VLM dependencies missing: {e}"
                )
                logger.warning(
                    "   Install: pip install "
                    "transformers>=4.52.0 accelerate "
                    "huggingface_hub bitsandbytes"
                )
            except Exception as e:
                logger.warning(
                    f"⚠️  VLM setup failed: {e}"
                )

        elif config.vlm_backend == "ollama":
            try:
                from ollama_vlm_report_generator import (
                    OllamaClient,
                )
                client = OllamaClient(config.ollama_vlm)

                if client.health_check():
                    state["ollama_ready"] = True
                    logger.info("✅ Ollama server reachable")
                else:
                    logger.warning(
                        f"⚠️  Cannot reach Ollama at "
                        f"{config.ollama_vlm.host}"
                    )
            except Exception as e:
                logger.warning(
                    f"⚠️  Ollama check failed: {e}"
                )

        # ── Startup summary ──
        logger.info("=" * 60)
        logger.info(
            f"  Pipeline ready:    {state['pipeline_ready']}"
        )
        logger.info(
            f"  VLM backend:       {state['vlm_backend']}"
        )
        if config.vlm_backend == "local":
            logger.info(
                f"  Model:             "
                f"{config.vlm.model_name}"
            )
            logger.info(
                f"  Model cached:      "
                f"{state['vlm_model_cached']}"
            )
            logger.info(
                f"  Model loaded:      "
                f"{state['vlm_model_loaded']}"
            )
            logger.info(
                f"  4-bit quantized:   "
                f"{config.vlm.load_in_4bit}"
            )
        else:
            logger.info(
                f"  Ollama ready:      {state['ollama_ready']}"
            )
        logger.info("=" * 60)

        if (
            state["pipeline_ready"]
            and state["vlm_model_loaded"]
        ):
            logger.info(
                "🚀 Server fully ready — accepting requests"
            )
        elif state["pipeline_ready"]:
            logger.info(
                "⚠️  Server ready for template/retrieval. "
                "VLM not loaded."
            )
        else:
            logger.info(
                "❌ Server NOT ready. Load pipeline first."
            )

        yield

        # Shutdown
        logger.info("Shutting down API server...")
        try:
            pipeline.unload_vlm()
        except Exception:
            pass
        logger.info("API server stopped.")

    # ── Create app ──
    app = FastAPI(
        title="Medical Image Retrieval API",
        description=(
            "API for medical image retrieval and radiology "
            "report generation.\n\n"
            "**VLM Backends:**\n"
            "- `local` — MedGemma 4B-IT "
            "(google/medgemma-4b-it)\n"
            "- `ollama` — Ollama remote VLM\n\n"
            "**Report Methods:**\n"
            "- `vlm_few_shot` — VLM with retrieved examples\n"
            "- `vlm_zero_shot` — VLM without examples\n"
            "- `template` — Template with weighted probs\n"
            "- `visual` — Side-by-side gallery\n"
            "- `weighted` / `majority` / `concat`\n"
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ──
    @app.middleware("http")
    async def add_timing(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        response.headers["X-Response-Time"] = f"{elapsed:.3f}s"

        if request.url.path.startswith("/api/"):
            state["request_count"] += 1
            logger.info(
                f"{request.method} {request.url.path} "
                f"→ {response.status_code} ({elapsed:.2f}s)"
            )

        return response

    # ── Helpers ──
    def _load_upload_image(
        upload_file: UploadFile,
    ) -> Image.Image:
        try:
            contents = upload_file.file.read()
            image = Image.open(
                io.BytesIO(contents)
            ).convert("RGB")
            return image
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {e}",
            )

    def _ensure_ready():
        if not state["pipeline_ready"]:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pipeline not ready. Load checkpoint "
                    "and index first via POST /api/load."
                ),
            )

    # ============================================================ #
    #  GET /api/health
    # ============================================================ #

    @app.get("/api/health")
    async def health_check():
        return {
            "status": "ok",
            "pipeline_ready": state["pipeline_ready"],
            "vlm_backend": state["vlm_backend"],
            "vlm_model_cached": state["vlm_model_cached"],
            "vlm_model_loaded": state["vlm_model_loaded"],
            "ollama_ready": state["ollama_ready"],
            "uptime_seconds": (
                (datetime.now() - state["startup_time"]).seconds
                if state["startup_time"]
                else 0
            ),
            "total_requests": state["request_count"],
            "timestamp": datetime.now().isoformat(),
        }

    # ============================================================ #
    #  GET /api/status
    # ============================================================ #

    @app.get("/api/status")
    async def get_status():
        status = {
            "pipeline_ready": state["pipeline_ready"],
            "vlm_backend": state["vlm_backend"],
            "vlm_model": config.vlm.model_name,
            "vlm_model_cached": state["vlm_model_cached"],
            "vlm_model_loaded": state["vlm_model_loaded"],
            "ollama_ready": state["ollama_ready"],
            "uptime_seconds": (
                (datetime.now() - state["startup_time"]).seconds
                if state["startup_time"]
                else 0
            ),
            "total_requests": state["request_count"],
            "avg_generation_time_s": (
                round(
                    state["total_generation_time"]
                    / max(state["request_count"], 1),
                    2,
                )
            ),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "vlm_backend": config.vlm_backend,
                "vlm_model": config.vlm.model_name,
                "vlm_dtype": config.vlm.torch_dtype,
                "vlm_4bit": config.vlm.load_in_4bit,
                "vlm_8bit": config.vlm.load_in_8bit,
                "vlm_cache_dir": config.vlm.local_model_dir,
                "vlm_max_tokens": config.vlm.max_new_tokens,
                "vlm_temperature": config.vlm.temperature,
                "vlm_num_examples": config.vlm.num_examples,
                "output_dir": config.output_dir,
            },
            "query_log": {
                "path": query_log.log_path,
                "entry_count": query_log.get_entry_count(),
            },
        }

        if config.vlm_backend == "ollama":
            status["config"]["ollama_host"] = (
                config.ollama_vlm.host
            )
            status["config"]["ollama_model"] = (
                config.ollama_vlm.model_name
            )

        if state["pipeline_ready"]:
            try:
                ib = pipeline._init_index_builder()
                stats = ib.get_index_stats()
                status["index"] = stats
            except Exception:
                status["index"] = {
                    "error": "Could not get index stats"
                }

        return status

    # ============================================================ #
    #  POST /api/load
    # ============================================================ #

    @app.post("/api/load")
    async def load_pipeline(
        checkpoint: str = Form(default="final_model"),
    ):
        try:
            pipeline.load_checkpoint(checkpoint)
            pipeline.load_index()
            state["pipeline_ready"] = True
            return {
                "status": "ok",
                "message": (
                    f"Loaded checkpoint '{checkpoint}' "
                    f"and FAISS index"
                ),
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load: {e}",
            )

    # ============================================================ #
    #  VLM Model Management (MedGemma)
    # ============================================================ #

    @app.get("/api/vlm/status")
    async def vlm_model_status():
        """Check MedGemma download and load status."""
        from vlm_report_generator import check_medgemma_status

        status = check_medgemma_status(
            model_name=config.vlm.model_name,
            cache_dir=config.vlm.local_model_dir,
        )
        status["model_in_memory"] = state["vlm_model_loaded"]
        status["vlm_backend"] = config.vlm_backend
        return status

    @app.post("/api/vlm/download")
    async def vlm_download():
        """
        Download MedGemma model to local cache (~8 GB).
        Blocks until complete. Only needed once.
        """
        from vlm_report_generator import download_medgemma

        try:
            path = download_medgemma(
                model_name=config.vlm.model_name,
                cache_dir=config.vlm.local_model_dir,
                hf_token=config.vlm.hf_token,
            )
            state["vlm_model_cached"] = True
            return {
                "status": "downloaded",
                "model": config.vlm.model_name,
                "path": str(path),
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Download failed: {e}",
            )

    @app.post("/api/vlm/load")
    async def vlm_load():
        """
        Pre-load MedGemma into GPU/CPU memory.
        Optional — auto-loads on first query.
        """
        try:
            from vlm_report_generator import (
                MedGemmaModelManager,
            )

            manager = MedGemmaModelManager.get_instance()
            manager.load_model(config.vlm)
            state["vlm_model_loaded"] = True
            state["vlm_model_cached"] = True

            return {
                "status": "loaded",
                "model": config.vlm.model_name,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Model load failed: {e}",
            )

    @app.post("/api/vlm/unload")
    async def vlm_unload():
        """Unload MedGemma from memory to free VRAM/RAM."""
        try:
            from vlm_report_generator import (
                MedGemmaModelManager,
            )

            manager = MedGemmaModelManager.get_instance()
            manager.unload_model()
            state["vlm_model_loaded"] = False

            return {"status": "unloaded"}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unload failed: {e}",
            )

    # ============================================================ #
    #  ENDPOINT 1: POST /api/retrieve
    # ============================================================ #

    @app.post("/api/retrieve")
    async def retrieve_similar(
        image: UploadFile = File(
            ..., description="Query medical image"
        ),
        top_k: int = Form(default=3, ge=1, le=20),
        text_query: Optional[str] = Form(default=None),
        use_query_expansion: bool = Form(default=True),
        use_reranking: bool = Form(default=True),
        min_score: float = Form(
            default=0.3, ge=0.0, le=1.0
        ),
    ):
        """Retrieve top-K similar images and their captions."""
        _ensure_ready()

        query_image = _load_upload_image(image)
        query_id = str(uuid.uuid4())[:8]

        logger.info(
            f"[{query_id}] Retrieve: top_k={top_k}, "
            f"min_score={min_score}"
        )

        try:
            re_ = pipeline._init_retrieval_engine()
            results = re_.retrieve_similar(
                query_image,
                top_k=top_k,
                text_query=text_query,
                use_query_expansion=use_query_expansion,
                use_reranking=use_reranking,
            )

            filtered = [
                r for r in results
                if r.get("score", 0) >= min_score
            ]
            if not filtered and results:
                filtered = [results[0]]

            response_results = []
            for i, r in enumerate(filtered):
                response_results.append({
                    "rank": r.get("rank", i + 1),
                    "score": round(r.get("score", 0.0), 6),
                    "caption": r.get("caption", ""),
                    "original_index": r.get(
                        "original_index", -1
                    ),
                    "image_available": (
                        r.get("original_index", -1) >= 0
                    ),
                })

            return {
                "query_id": query_id,
                "num_results": len(response_results),
                "top_k_requested": top_k,
                "min_score": min_score,
                "results": response_results,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(
                f"[{query_id}] Retrieval failed: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Retrieval failed: {e}",
            )

    # ============================================================ #
    #  ENDPOINT 2: POST /api/generate-report
    # ============================================================ #

    @app.post("/api/generate-report")
    async def generate_report(
        image: UploadFile = File(
            ..., description="Query medical image"
        ),
        report_method: str = Form(
            default="vlm_few_shot",
            description=(
                "Report method: vlm_few_shot, vlm_zero_shot, "
                "template, visual, weighted, majority, concat"
            ),
        ),
        top_k: int = Form(default=3, ge=1, le=10),
        min_score: float = Form(
            default=0.3, ge=0.0, le=1.0
        ),
        temperature: Optional[float] = Form(default=None),
        max_tokens: Optional[int] = Form(default=None),
        num_examples: Optional[int] = Form(default=None),
    ):
        """
        Generate a medical report from a query image.

        report_method options:
          - "vlm_few_shot"   — MedGemma with retrieved examples
          - "vlm_zero_shot"  — MedGemma without examples
          - "template"       — Template with weighted probs
          - "visual"         — Side-by-side gallery
          - "weighted"       — Weighted retrieval report
          - "majority"       — Majority vote report
          - "concat"         — Concatenated captions

        For Ollama, use "ollama_few_shot" / "ollama_zero_shot"
        """
        _ensure_ready()

        query_image = _load_upload_image(image)
        query_id = str(uuid.uuid4())[:8]

        valid_methods = [
            "vlm_few_shot", "vlm_zero_shot",
            "ollama_few_shot", "ollama_zero_shot",
            "template", "visual",
            "weighted", "majority", "concat",
        ]
        if report_method not in valid_methods:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid report_method '{report_method}'. "
                    f"Valid: {valid_methods}"
                ),
            )

        logger.info(
            f"[{query_id}] Report: method={report_method}, "
            f"top_k={top_k}"
        )

        try:
            # Retrieve similar cases
            re_ = pipeline._init_retrieval_engine()
            retrieval_results = re_.retrieve_similar(
                query_image,
                top_k=top_k,
                use_query_expansion=True,
                use_reranking=True,
            )

            filtered_results = [
                r for r in retrieval_results
                if r.get("score", 0) >= min_score
            ]
            if not filtered_results and retrieval_results:
                filtered_results = [retrieval_results[0]]

            # Apply temp overrides
            orig_temp = config.vlm.temperature
            orig_max = config.vlm.max_new_tokens
            orig_ex = config.vlm.num_examples

            if temperature is not None:
                config.vlm.temperature = temperature
            if max_tokens is not None:
                config.vlm.max_new_tokens = max_tokens
            if num_examples is not None:
                config.vlm.num_examples = num_examples

            # Generate report
            rg = pipeline._init_report_generator()

            gen_start = time.time()
            report_output = rg.generate_report(
                filtered_results,
                method=report_method,
                query_image=query_image,
            )
            gen_elapsed = time.time() - gen_start

            # Restore config
            config.vlm.temperature = orig_temp
            config.vlm.max_new_tokens = orig_max
            config.vlm.num_examples = orig_ex

            # Update state
            state["total_generation_time"] += gen_elapsed

            if report_method.startswith("vlm_"):
                state["vlm_model_loaded"] = True
                state["vlm_model_cached"] = True

            # Build response
            retrieval_summary = [
                {
                    "rank": r.get("rank", i + 1),
                    "score": round(r.get("score", 0), 6),
                    "caption": r.get("caption", ""),
                    "original_index": r.get(
                        "original_index", -1
                    ),
                }
                for i, r in enumerate(filtered_results)
            ]

            logger.info(
                f"[{query_id}] Report generated in "
                f"{gen_elapsed:.1f}s | "
                f"success={report_output.get('success')}"
            )

            return {
                "query_id": query_id,
                "report": report_output.get("report", ""),
                "raw_vlm_output": report_output.get(
                    "raw_vlm_output", ""
                ),
                "method": report_output.get(
                    "method", report_method
                ),
                "model": report_output.get(
                    "model", config.vlm.model_name
                ),
                "backend": report_output.get(
                    "backend", config.vlm_backend
                ),
                "num_examples": report_output.get(
                    "num_examples", 0
                ),
                "num_images_sent": report_output.get(
                    "num_images_sent", 0
                ),
                "retrieval_results": retrieval_summary,
                "detected_conditions": report_output.get(
                    "detected_conditions"
                ),
                "success": report_output.get("success", False),
                "error": report_output.get("error"),
                "generation_time_s": round(gen_elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(
                f"[{query_id}] Report failed: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Report generation failed: {e}",
            )

    # ============================================================ #
    #  ENDPOINT 3: POST /api/index/append
    # ============================================================ #

    @app.post("/api/index/append")
    async def append_to_index(
        data_folder: str = Form(
            ...,
            description="Path to folder with CSV + images",
        ),
        rebuild_after: bool = Form(default=True),
    ):
        """Append new image-caption pairs to FAISS index."""
        _ensure_ready()

        if not os.path.isdir(data_folder):
            raise HTTPException(
                status_code=400,
                detail=f"Folder not found: {data_folder}",
            )

        try:
            pairs = CSVDataLoader.load_from_folder(data_folder)

            if not pairs:
                return {
                    "status": "warning",
                    "message": "No valid pairs found",
                    "new_entries": 0,
                }

            logger.info(
                f"Appending {len(pairs)} entries to index..."
            )

            import torch
            import faiss

            ft = pipeline._init_finetuner()
            ib = pipeline._init_index_builder()

            added = 0
            skipped = 0

            for pair in pairs:
                try:
                    img = Image.open(
                        pair["image_path"]
                    ).convert("RGB")
                    caption = pair["caption"]

                    processed = ft.processor(
                        images=img, return_tensors="pt",
                    )
                    pixel_values = processed[
                        "pixel_values"
                    ].to(ft.device)

                    with torch.no_grad():
                        embedding = ft.encode_image(
                            pixel_values
                        )

                    emb_np = embedding.cpu().numpy()
                    emb_transformed = (
                        ib.transform_query_embedding(emb_np)
                    )

                    current_size = len(ib.metadata)

                    emb_add = emb_transformed.astype(
                        np.float32
                    )
                    if config.faiss.normalize_embeddings:
                        faiss.normalize_L2(emb_add)

                    ib.index.add(emb_add)

                    ib.metadata.append({
                        "original_index": current_size,
                        "caption": caption,
                        "source": "api_append",
                        "image_path": pair["image_path"],
                        "added_at": datetime.now().isoformat(),
                    })

                    if ib.embeddings is not None:
                        ib.embeddings = np.vstack([
                            ib.embeddings, emb_add,
                        ])
                    else:
                        ib.embeddings = emb_add.copy()

                    added += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to add "
                        f"{pair['image_path']}: {e}"
                    )
                    skipped += 1

            if rebuild_after and added > 0:
                ib.save_index()

            total_size = ib.index.ntotal if ib.index else 0

            return {
                "status": "ok",
                "new_entries": added,
                "skipped": skipped,
                "total_index_size": total_size,
                "data_folder": data_folder,
                "timestamp": datetime.now().isoformat(),
            }

        except FileNotFoundError as e:
            raise HTTPException(
                status_code=404, detail=str(e)
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Index append failed: {e}",
            )

    # ============================================================ #
    #  ENDPOINT 4: POST /api/feedback
    # ============================================================ #

    @app.post("/api/feedback")
    async def submit_feedback(
        image: UploadFile = File(
            ..., description="The query image"
        ),
        feedback: str = Form(
            ..., description="User feedback text"
        ),
        generated_caption: str = Form(
            default="",
            description="Caption to store in index",
        ),
        query_id: Optional[str] = Form(default=None),
        report_method: Optional[str] = Form(default=None),
        add_to_index: bool = Form(default=True),
    ):
        """Submit feedback and optionally add to FAISS index."""
        query_image = _load_upload_image(image)

        if query_id is None:
            query_id = str(uuid.uuid4())[:8]

        caption_for_index = (
            generated_caption.strip()
            if generated_caption.strip()
            else feedback.strip()
        )

        logger.info(
            f"[{query_id}] Feedback: {len(feedback)} chars, "
            f"add_to_index={add_to_index}"
        )

        log_entry = query_log.add_entry(
            query_id=query_id,
            image_filename=image.filename or "unknown",
            generated_caption=caption_for_index,
            feedback=feedback,
            report_method=report_method,
            extra_metadata={
                "add_to_index": add_to_index,
            },
        )

        added_to_index = False
        index_size = 0

        if add_to_index and state["pipeline_ready"]:
            try:
                import torch
                import faiss

                ft = pipeline._init_finetuner()
                ib = pipeline._init_index_builder()

                processed = ft.processor(
                    images=query_image, return_tensors="pt",
                )
                pixel_values = processed[
                    "pixel_values"
                ].to(ft.device)

                with torch.no_grad():
                    embedding = ft.encode_image(pixel_values)

                emb_np = embedding.cpu().numpy()
                emb_transformed = (
                    ib.transform_query_embedding(emb_np)
                )

                current_size = len(ib.metadata)

                emb_add = emb_transformed.astype(np.float32)
                if config.faiss.normalize_embeddings:
                    faiss.normalize_L2(emb_add)

                ib.index.add(emb_add)

                ib.metadata.append({
                    "original_index": current_size,
                    "caption": caption_for_index,
                    "source": "user_feedback",
                    "feedback": feedback,
                    "query_id": query_id,
                    "added_at": datetime.now().isoformat(),
                })

                if ib.embeddings is not None:
                    ib.embeddings = np.vstack([
                        ib.embeddings, emb_add,
                    ])
                else:
                    ib.embeddings = emb_add.copy()

                ib.save_index()

                added_to_index = True
                index_size = ib.index.ntotal

            except Exception as e:
                logger.error(
                    f"[{query_id}] Index add failed: {e}",
                    exc_info=True,
                )

        return {
            "status": "ok",
            "query_id": query_id,
            "log_entry": log_entry,
            "added_to_index": added_to_index,
            "index_size": index_size,
            "log_total_entries": query_log.get_entry_count(),
            "timestamp": datetime.now().isoformat(),
        }

    # ============================================================ #
    #  GET /api/feedback/log
    # ============================================================ #

    @app.get("/api/feedback/log")
    async def get_feedback_log(
        date: Optional[str] = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        if date:
            entries = query_log.get_entries_by_date(date)
        else:
            entries = query_log.get_all_entries()

        entries = list(reversed(entries))[:limit]

        return {
            "total_entries": query_log.get_entry_count(),
            "returned": len(entries),
            "filter_date": date,
            "entries": entries,
        }

    # ============================================================ #
    #  POST /api/index/build
    # ============================================================ #

    @app.post("/api/index/build")
    async def build_index():
        try:
            pipeline.load_checkpoint("final_model")
            stats = pipeline.build_index()
            state["pipeline_ready"] = True
            return {
                "status": "ok",
                "message": "FAISS index built",
                "stats": stats,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Index build failed: {e}",
            )

    # ============================================================ #
    #  GET /api/gallery/{filename}
    # ============================================================ #

    @app.get("/api/gallery/{filename}")
    async def get_gallery_image(filename: str):
        file_path = os.path.join(config.output_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {filename}",
            )
        return FileResponse(
            file_path, media_type="image/png",
        )

    @app.get("/images/case/{original_index}")
    async def get_case_image(original_index: int):
        """Serve a dataset image by its original index."""
        try:
            from datasets import load_dataset
            # Use the already-cached dataset
            ds = load_dataset(
                "itsanmolgupta/mimic-cxr-dataset",
                cache_dir=config.cache_dir if hasattr(config, 'cache_dir') else "./cache",
                split="train",
            )
            item = ds[original_index]
            img = item.get("image")
            if img is None:
                raise ValueError("No image")
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert("RGB")
            else:
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            from fastapi.responses import StreamingResponse
            return StreamingResponse(buf, media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    return app


# ================================================================== #
#  Main
# ================================================================== #

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Medical Image Retrieval API Server",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
    )
    parser.add_argument(
        "--checkpoint", type=str, default="final_model",
    )
    parser.add_argument(
        "--no_auto_load", action="store_true",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
    )

    # VLM backend selection
    parser.add_argument(
        "--vlm_backend",
        choices=["local", "ollama"],
        default="local",
        help="VLM backend: local (MedGemma) or ollama",
    )

    # Local MedGemma settings
    parser.add_argument(
        "--vlm_model", type=str,
        default="google/medgemma-4b-it",
        help="Local VLM model name",
    )
    parser.add_argument(
        "--vlm_cache_dir", type=str,
        default="./cache/huggingface",
        help="Directory to download/cache VLM model",
    )
    parser.add_argument(
        "--vlm_4bit", action="store_true",
        help="Load MedGemma in 4-bit quantization",
    )
    parser.add_argument(
        "--vlm_8bit", action="store_true",
        help="Load MedGemma in 8-bit quantization",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token for gated models",
    )
    parser.add_argument(
        "--vlm_temperature", type=float, default=0.3,
    )
    parser.add_argument(
        "--vlm_max_tokens", type=int, default=1024,
    )
    parser.add_argument(
        "--vlm_num_examples", type=int, default=3,
    )
    parser.add_argument(
        "--vlm_dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for VLM model",
    )

    # Ollama settings (optional)
    parser.add_argument(
        "--ollama_host", type=str,
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--ollama_model", type=str,
        default="llava-llama3:8b",
    )

    # Server settings
    parser.add_argument(
        "--workers", type=int, default=1,
    )
    parser.add_argument(
        "--reload", action="store_true",
    )

    args = parser.parse_args()

    api_config = PipelineConfig(
        vlm=VLMConfig(
            model_name=args.vlm_model,
            torch_dtype=args.vlm_dtype,
            temperature=args.vlm_temperature,
            max_new_tokens=args.vlm_max_tokens,
            num_examples=args.vlm_num_examples,
            load_in_4bit=args.vlm_4bit,
            load_in_8bit=args.vlm_8bit,
            hf_token=args.hf_token,
            local_model_dir=args.vlm_cache_dir,
            enabled=True,
        ),
        ollama_vlm=OllamaVLMConfig(
            host=args.ollama_host,
            model_name=args.ollama_model,
        ),
        vlm_backend=args.vlm_backend,
        output_dir=args.output_dir,
    )

    app = create_app(
        config=api_config,
        checkpoint_name=args.checkpoint,
        auto_load=not args.no_auto_load,
    )

    logger.info(f"Starting API on {args.host}:{args.port}")
    logger.info(f"  VLM backend: {args.vlm_backend}")
    logger.info(f"  VLM model:   {args.vlm_model}")
    logger.info(f"  VLM cache:   {args.vlm_cache_dir}")
    logger.info(f"  VLM 4-bit:   {args.vlm_4bit}")
    logger.info(f"  Docs:  http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()