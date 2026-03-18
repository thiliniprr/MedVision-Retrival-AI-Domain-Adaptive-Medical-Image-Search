# ====================
# MedVision Backend
# ====================
#
# Project structure this file expects:
#
#   parent/
#   ├── backend.py               ← this file
#   ├── Finetuning_pipeline/
#   │   ├── main_pipeline.py
#   │   ├── config.py
#   │   ├── clip_finetuner.py
#   │   └── ...
#   └── medvision_frontend/
#       └── app.py
#
# ── How to run (always from the parent/ folder) ───────────────────────────────
#
#   # 1. Install dependencies (once)
#   pip install fastapi uvicorn python-multipart
#
#   # 2. Start the backend
#   cd parent/
#   uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
#
#   # 3. Start the frontend (new terminal, same parent/ folder)
#   streamlit run medvision_frontend/app.py
#
# ─────────────────────────────────────────────────────────────────────────────

import sys
import uuid
import shutil
import io
import numpy as np
from PIL import Image
from fastapi.responses import StreamingResponse
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Make sure Python can find Finetuning_pipeline/ ───────────────────────────
# This adds the parent/ folder to sys.path so the imports below work
# regardless of how you launch uvicorn.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Import your teammate's pipeline ──────────────────────────────────────────
from Finetuning_pipeline.main_pipeline import MedicalImageRetrievalPipeline
from Finetuning_pipeline.config import (
    PipelineConfig, ModelConfig, DatasetConfig,
    FineTuneConfig, FAISSConfig, RetrievalConfig,
)

# ── Upload directory (created next to backend.py in parent/) ─────────────────
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedVision API",
    description="Medical image retrieval and radiology report generation.",
    version="1.0.0",
)

# Allows the Streamlit frontend on a different port to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load pipeline once at startup ─────────────────────────────────────────────
config = PipelineConfig(
    model=ModelConfig(),
    dataset=DatasetConfig(),
    finetune=FineTuneConfig(
        fp16=False,                     # disable fp16 — not supported on CPU
        projection_dim=195,               
    ),
    faiss=FAISSConfig(
        index_save_path=r"C:\Users\allir\Documents\GitHub\SOLO_NextGen_Case-3-MedVision-Retrival-AI-Domain-Adaptive-Medical-Image-Search\Finetuning_pipeline\faiss_index",
    ),
    retrieval=RetrievalConfig(top_k=5),
    device="cpu",                 # change to "cuda" if you have a GPU
    cache_dir=r"C:\Users\allir\Documents\GitHub\SOLO_NextGen_Case-3-MedVision-Retrival-AI-Domain-Adaptive-Medical-Image-Search\Finetuning_pipeline\cache",
    checkpoint_dir=r"C:\Users\allir\Documents\GitHub\SOLO_NextGen_Case-3-MedVision-Retrival-AI-Domain-Adaptive-Medical-Image-Search\Finetuning_pipeline\checkpoints",
    output_dir=r"C:\Users\allir\Documents\GitHub\SOLO_NextGen_Case-3-MedVision-Retrival-AI-Domain-Adaptive-Medical-Image-Search\Finetuning_pipeline\output",
)

# 
config.retrieval.use_query_expansion = False
config.retrieval.use_reranking = False
config.retrieval.use_multimodal_search = False

pipeline = MedicalImageRetrievalPipeline(config)

try:
    pipeline.load_checkpoint("final_model")
    pipeline.load_index()
    print("✓ Checkpoint and FAISS index loaded.")
except Exception as e:
    print(f"⚠  Could not load checkpoint/index at startup: {e}")
    print("   Call POST /pipeline/load before running queries.")

# ── Load dataset for image serving ───────────────────────────────────────────
dataset = None
try:
    from datasets import load_dataset
    dataset = load_dataset(
        "itsanmolgupta/mimic-cxr-dataset",
        cache_dir=r"C:\Users\allir\Documents\GitHub\SOLO_NextGen_Case-3-MedVision-Retrival-AI-Domain-Adaptive-Medical-Image-Search\Finetuning_pipeline\cache",
        split="train",
    )
    print(f"✓ Dataset loaded for image serving ({len(dataset)} images).")
except Exception as e:
    print(f"⚠  Dataset not loaded — case images won't be available: {e}")

# ── In-memory map: image_id → absolute file path ─────────────────────────────
image_store: dict[str, Path] = {}


# ══════════════════════════════════════════════════════════════════════════════
#  Request / Response models
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    image_id: str
    top_k: Optional[int] = 5
    report_method: Optional[str] = "template"  # template | weighted | majority | concat

class LoadRequest(BaseModel):
    checkpoint: Optional[str] = "best_model"


# ══════════════════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """Liveness check — Streamlit polls this to show the status badge."""
    return {"status": "ok"}

@app.get("/images/case/{original_index}")
def get_case_image(original_index: int):
    """Return a dataset image by its original index as JPEG."""
    if dataset is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")
    try:
        item = dataset[original_index]
        img  = item.get("image")

        if img is None:
            raise ValueError("No image at this index")

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        else:
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load image {original_index}: {e}")


@app.post("/images/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Saves the uploaded X-ray to parent/uploads/ and returns an image_id.
    The frontend stores this ID and sends it back when requesting retrieval.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")

    image_id = str(uuid.uuid4())
    suffix   = Path(file.filename).suffix or ".jpg"
    dest     = UPLOAD_DIR / f"{image_id}{suffix}"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    image_store[image_id] = dest
    return {"image_id": image_id, "filename": file.filename}


@app.post("/retrieval/query")
def query(req: QueryRequest):
    """
    Looks up the image by image_id, runs pipeline.query(), and returns
    the generated report + similar cases to the frontend.
    """
    if req.image_id not in image_store:
        raise HTTPException(
            status_code=404,
            detail=f"image_id '{req.image_id}' not found. Upload the image first."
        )

    image_path = image_store[req.image_id]

    try:
        result = pipeline.query(
            image_path    = str(image_path),
            top_k         = req.top_k,
            report_method = req.report_method,
            use_query_expansion = False,   
            use_reranking       = False,
            use_multimodal      = False,   
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    result["method"] = req.report_method
    return result


@app.post("/pipeline/load")
def load_pipeline_checkpoint(req: LoadRequest):
    """
    Reload the checkpoint + FAISS index at runtime.
    Call this once after training finishes if you skipped loading at startup.
    """
    try:
        pipeline.load_checkpoint(req.checkpoint)
        pipeline.load_index()
        return {"status": "loaded", "checkpoint": req.checkpoint}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/finetune")
def finetune():
    """Trigger fine-tuning. Blocks until complete."""
    try:
        history = pipeline.finetune()
        return {"status": "complete", "epochs": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/build-index")
def build_index():
    """Re-encode the training corpus and rebuild the FAISS index."""
    try:
        stats = pipeline.build_index()
        return {"status": "complete", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
