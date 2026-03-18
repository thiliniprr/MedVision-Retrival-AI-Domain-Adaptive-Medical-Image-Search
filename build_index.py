from Finetuning_pipeline.main_pipeline import MedicalImageRetrievalPipeline
from Finetuning_pipeline.config import PipelineConfig

# Load config and pipeline
config = PipelineConfig()
pipeline = MedicalImageRetrievalPipeline(config)

# Build and save the FAISS index
pipeline.build_index()
pipeline.save_index()

print("FAISS index built and saved successfully.")
