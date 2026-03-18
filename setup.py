# ====================
# MedVision Setup
# ====================
# Run this ONCE from the parent/ folder before starting the backend.
#
#   cd parent/
#   python setup.py
#
# This will:
#   1. Fine-tune CLIP on MIMIC-CXR   (~hours on CPU, ~30min on GPU)
#   2. Build the FAISS index          (~minutes)
#   3. Save both to disk so FastAPI_backend.py can load them on startup
#
# After this completes, start the backend normally:
#   uvicorn FastAPI_backend:app --host 0.0.0.0 --port 8000 --reload

import sys
import time
import multiprocessing
from pathlib import Path

# Make sure Finetuning_pipeline/ is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from Finetuning_pipeline.main_pipeline import MedicalImageRetrievalPipeline
from Finetuning_pipeline.config import (
    PipelineConfig, ModelConfig, DatasetConfig,
    FineTuneConfig, FAISSConfig, RetrievalConfig,
)

# ── Required on Windows ───────────────────────────────────────────────────────
# DataLoader with num_workers > 0 spawns child processes on Windows.
# Without this guard, each child re-runs the whole script and crashes.
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # ── Configure ─────────────────────────────────────────────────────────────
    config = PipelineConfig(
        model=ModelConfig(),
        dataset=DatasetConfig(
            max_train_samples=500,),   # None; set up to 500 for a quick smoke-test
        finetune=FineTuneConfig(
            batch_size=16,            # lower to 8 if you run out of memory
            num_epochs=1,               # 10 for real fine-tuning; set to 1 for a quick smoke-test
            learning_rate=5e-6,
            fp16=False,               # not supported on CPU
        ),
        faiss=FAISSConfig(
            index_save_path="C:/medvision/faiss_index",
        ),
        retrieval=RetrievalConfig(top_k=5),
        device="cpu",                 # change to "cuda" if you have a GPU
        cache_dir="C:/medvision/cache",
        checkpoint_dir="C:/medvision/checkpoints",
        output_dir="C:/medvision/output",
    )

    pipeline = MedicalImageRetrievalPipeline(config)

    # ── Stage 1: Fine-tune ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1 — Fine-tuning CLIP on MIMIC-CXR")
    print("="*60)
    t0 = time.time()
    history = pipeline.finetune()
    print(f"Fine-tuning done in {(time.time()-t0)/60:.1f} min")
    print(f"Final train loss: {history[-1]['train_loss']:.4f}" if history else "")

    # ── Stage 2: Build FAISS index ────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2 — Building FAISS index")
    print("="*60)
    t0 = time.time()
    stats = pipeline.build_index()
    print(f"Index built in {(time.time()-t0)/60:.1f} min")
    print(f"Index stats: {stats}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("✓ Setup complete!")
    print("="*60)
    print("\nYou can now start the backend:")
    print("  uvicorn backend:app --host 0.0.0.0 --port 8000 --reload")
    print("\nAnd the frontend (new terminal):")
    print("  streamlit run medvision_frontend/app.py")
