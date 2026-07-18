python main_pipeline.py --mode status

# Full pipeline with ALL enhancements (default)
python main_pipeline.py --mode full \
    --epochs 10 \
    --batch_size 32 \
    --query_image ./test_xray.png

# Full pipeline with limited training data (for testing)
python main_pipeline.py --mode full \
    --epochs 5 \
    --batch_size 16 \
    --max_train_samples 1000 \
    --query_image ./test_xray.png
    
    
    
# Train without hard negatives (faster, less accurate)
python main_pipeline.py --mode full \
    --no_hard_negatives \
    --epochs 10

# Build index without whitening
python main_pipeline.py --mode index \
    --no_whitening

# Query without expansion or re-ranking (fastest, baseline)
python main_pipeline.py --mode query \
    --query_image ./test_xray.png \
    --no_query_expansion \
    --no_reranking

# Use simple projection head (original behavior)
python main_pipeline.py --mode full \
    --simple_projection
    
    
#multimodal query
python main_pipeline.py --mode query \
    --query_image ./test_xray.png \
    --use_multimodal \
    --text_query "bilateral pleural effusion with cardiomegaly" \
    --top_k 10
    

    
# Stage 1: Fine-tune only
python main_pipeline.py --mode finetune --epochs 10

# Stage 2: Build index from existing checkpoint
python main_pipeline.py --mode index --checkpoint best_model

# Stage 3: Query with existing model + index
python main_pipeline.py --mode query \
    --query_image ./test_xray.png \
    --report_method template \
    --top_k 5
    
    
# Skip training, just build index and query
python main_pipeline.py --mode full \
    --skip_finetune \
    --query_image ./test_xray.png

# Skip both training and indexing, just query
python main_pipeline.py --mode full \
    --skip_finetune \
    --skip_index \
    --query_image ./test_xray.png
    
    
#gradio
python app.py
# Opens at http://localhost:7860


# Few-shot VLM report (retrieves 3 examples, feeds to VLM)
python main_pipeline.py --mode query_vlm \
    --query_image ./test_xray.png \
    --vlm_mode few_shot \
    --top_k 3 \
    --vlm_temperature 0.3

# Zero-shot VLM report (no examples)
python main_pipeline.py --mode query_vlm \
    --query_image ./test_xray.png \
    --vlm_mode zero_shot

# Few-shot with custom model and more examples
python main_pipeline.py --mode query_vlm \
    --query_image ./test_xray.png \
    --vlm_mode few_shot \
    --top_k 5 \
    --vlm_num_examples 5 \
    --vlm_model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --vlm_max_tokens 768

# Standard query but using VLM report method
python main_pipeline.py --mode query \
    --query_image ./test_xray.png \
    --report_method vlm_few_shot \
    --top_k 5

# Disable VLM entirely
python main_pipeline.py --mode full \
    --no_vlm \
    --query_image ./test_xray.png