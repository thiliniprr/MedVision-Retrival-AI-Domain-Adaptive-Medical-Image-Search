# Medical Image Retrieval & Report Generation System

A comprehensive pipeline for **medical image similarity retrieval** and **automated radiology report generation** built on CLIP fine-tuning, FAISS indexing, and Vision-Language Model (VLM) inference.

The system fine-tunes OpenAI's CLIP on the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) chest X-ray dataset, builds a high-performance vector index for similarity search, and generates structured radiology reports using retrieval-augmented generation — either via template-based methods or through **llava-llama3:8b** served by [Ollama](https://ollama.com/).

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                               │
│                                                                     │
│  MIMIC-CXR Dataset ──► CLIP Fine-Tuning ──► Medical CLIP Model      │
│  (images + reports)     (contrastive loss)   (domain-adapted)       │
│                         + Hard Negatives                            │
│                         + Improved Projections                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        INDEXING PHASE                               │
│                                                                     │
│  Medical CLIP ──► Encode All Images ──► PCA Whitening ──► FAISS     │
│                   (embed dataset)       (post-process)    Index     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                              │
│                                                                     │
│  Query Image ──► Encode ──► (Query Expansion) ──► FAISS Search      │
│                             (Rocchio feedback)    (retrieve top-K)  │
│                                                        │            │
│                                    ┌───────────────────┤            │
│                                    ▼                   ▼            │
│                             Re-Ranking           Report Generation  │
│                          (cosine refine)    ┌──────────┼──────────┐ │
│                                             ▼          ▼          ▼ │
│                                         Template    Visual       VLM│
│                                         Report      Gallery   Report│
│                                                        ┌──────────┤ │
│                                                        ▼          ▼ │
│                                                      Local   Ollama │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation

### 1. Clone & Install Dependencies

```bash
git clone <repository-url>
cd SOLO_NextGen_Case-3-MedVision-Retrival-AI-Domain-Adaptive-Medical-Image-Search

pip install -r requirement.txt
```

**Core dependencies:**

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework |
| `transformers` | CLIP model & tokenizers |
| `datasets` | HuggingFace MIMIC-CXR loading |
| `faiss-cpu` / `faiss-gpu` | Vector similarity search |
| `Pillow` | Image processing |
| `matplotlib` | Visual report generation |
| `requests` | Ollama HTTP communication |
| `numpy`, `tqdm`, `logging` | Utilities |

### 2. Set Up Ollama for VLM Report Generation

On the **Ollama server machine** (can be the same or a remote GPU machine):

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the llava-llama3:8b model (~5GB)
ollama pull llava-llama3:8b

# Start Ollama with network access (for remote use)
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

> **Note:** If running Ollama on a remote machine, ensure port `11434` is open in the firewall:
> ```bash
> sudo ufw allow 11434/tcp
> ```

---

## 🚀 Usage

The system operates in three sequential stages. Each stage can be run independently or together via the CLI.

---

### Stage 1: Fine-Tune CLIP on Medical Data

Fine-tunes a pretrained CLIP model on MIMIC-CXR image–report pairs using symmetric contrastive learning. The fine-tuning adapts CLIP's vision and text encoders to the medical imaging domain.

**What happens:**
1. Loads the MIMIC-CXR dataset from HuggingFace
2. Combines `findings` and `impression` text columns with smart truncation (CLIP's 77-token limit)
3. Trains with contrastive loss (InfoNCE) + optional hard negative mining
4. Applies improved multi-layer projection heads for better embedding quality
5. Saves checkpoints to `./checkpoints/`

```bash
# Basic fine-tuning
python main_pipeline.py --mode finetune \
    --epochs 10 \
    --batch_size 32 \
    --lr 5e-6

# Fine-tuning with simpler projection (less GPU memory)
python main_pipeline.py --mode finetune \
    --epochs 10 \
    --batch_size 64 \
    --simple_projection

# Limit training data (for quick experiments)
python main_pipeline.py --mode finetune \
    --epochs 5 \
    --max_train_samples 5000
```

---

### Stage 2: Build the FAISS Index

Encodes all dataset images through the fine-tuned CLIP model and builds a FAISS vector index for fast similarity search.

**What happens:**
1. Loads the fine-tuned CLIP checkpoint
2. Encodes every image in the dataset into a dense embedding vector
3. Applies PCA whitening to decorrelate dimensions and improve cosine similarity
4. Builds a FAISS index (default: `IndexFlatIP` for exact inner product search)
5. Saves the index, embeddings, metadata, and preprocessing parameters to `./faiss-index/`

```bash
# Build index from the final fine-tuned model
python main_pipeline.py --mode index \
    --checkpoint final_model

# Build index without whitening
python main_pipeline.py --mode index \
    --checkpoint final_model \
    --no_whitening
```
---

### Stage 3: Report Generation

Given a query chest X-ray image, the system retrieves the most similar cases from the FAISS index and generates a radiology report. Three report generation methods are supported:

---

#### Method 1: Template-Based Report (`template`)

Generates a structured radiology report by analyzing the captions of retrieved similar cases and mapping detected conditions to a standard report template.

**How it works:**
1. Retrieve top-K similar cases from the index
2. Scan all retrieved captions for medical condition keywords (cardiomegaly, pneumonia, effusion, etc.)
3. Compute frequency of each condition across retrieved cases
4. Fill a structured FINDINGS + IMPRESSION template based on detected condition frequencies (≥30% threshold)

```bash
python main_pipeline.py \
    --mode query_template \
    --query_image /path/to/chest_xray.jpg \
    --checkpoint final_model \
    --top_k 3
```

**Example output:**

```
=================================================================
  RADIOLOGY REPORT (Template-Based)
  Based on 3 similar cases (score >= 0.3)
=================================================================

REFERENCE CASES:
  [1] Score: 0.9993
      No acute cardiopulmonary process. The lungs are clear of focal consolidation, pleural effusion or pneumothorax. The heart size is normal. The mediastinal contours are normal. Multiple surgical clips p

  [2] Score: 0.3135
      No acute cardiopulmonary process. The lungs are clear. There is no effusion, consolidation, or edema. The cardiomediastinal silhouette is within normal limits. No acute osseous abnormalities identifie

  [3] Score: 0.3088
      No acute cardiopulmonary process. No dispalced fracture is identified. If there is concern for rib fracture, dedicated rib films can be obtained. ap view of the chest . there is no focal consolidation

FINDINGS (with weighted probability):
  Heart: Normal cardiac silhouette. [P(cardiomegaly) ≈ 0%]
  Lungs: Airspace opacity suggesting pneumonia. [P = 100.0%]
  Lungs: Pulmonary edema. [P = 19.3%]
  Pleura: Pleural effusion noted. [P = 100.0%]
  Bones: Fracture suspected. [P = 80.7%]

IMPRESSION:
  1. Pleural Effusion — weighted probability: 100.0%
  2. Pneumonia — weighted probability: 100.0%
  3. Pneumothorax — weighted probability: 80.7%
  4. Fracture — weighted probability: 80.7%
  5. Edema — weighted probability: 19.3%

-----------------------------------------------------------------
  Based on 3 similar cases (scores: 0.9993, 0.3135, 0.3088)
  Score threshold: >= 0.3
=================================================================
```

---

#### Method 2: Visual Gallery Report (`visual`)

Creates a visual comparison figure showing the query image alongside the top-K retrieved similar cases with similarity scores, color-coded borders, and captions.

**How it works:**
1. Retrieve top-K similar cases
2. Load the actual images from the dataset for each retrieved case
3. Generate a matplotlib figure or pure-PIL gallery showing:
   - Query image (highlighted with blue border)
   - Retrieved images with color-coded similarity borders (green ≥ 0.8, orange ≥ 0.6, red < 0.4)
   - Similarity scores and truncated captions
4. Optionally generate a detailed report with a conditions summary panel
5. Save to `./output/`

```bash
python main_pipeline.py \
    --mode query_template \
    --query_image /path/to/chest_xray.jpg \
    --checkpoint final_model \
    --top_k 3
```

**Generated files:**
- `./output/visual_gallery.png` — Side-by-side image comparison
- `./output/detailed_visual_report.png` — Extended report with conditions panel

**Visual layout:**

```
<img width="1425" height="1390" alt="image" src="https://github.com/user-attachments/assets/9368c5c5-5404-4e12-a6c6-093742c288de" />


```

---

#### Method 3: Ollama VLM Report (`ollama_few_shot`)

Generates a comprehensive radiology report using **llava-llama3:8b** served via Ollama. The model receives the query image along with captions from retrieved similar cases as few-shot context, producing a detailed FINDINGS + IMPRESSION report.

**How it works:**
1. Retrieve top-K similar cases and their captions
2. Detect conditions from retrieved captions (used as supplementary context)
3. Build a structured prompt with:
   - System prompt (expert radiologist persona)
   - Retrieved captions as numbered reference examples with similarity scores
   - Detected conditions summary
   - Output format instructions (FINDINGS + IMPRESSION)
4. Encode the query image to base64
5. Send prompt + image to Ollama `/api/chat` endpoint
6. llava-llama3:8b generates a detailed medical report
7. Clean and format the output

**Prerequisites:**
```bash
# Verify Ollama connection first
python main_pipeline.py --mode check_ollama \
    --ollama_host http://<server-ip>:11434
```

**Few-shot generation** (recommended — uses retrieved captions as context):
```bash
python main_pipeline.py --mode query_ollama \
    --query_image /path/to/chest_xray.jpg \
    --checkpoint final_model \
    --top_k 5 \
    --vlm_mode few_shot \
    --ollama_host http://localhost:11434 \
    --ollama_model llava-llama3:8b \
    --vlm_temperature 0.3 \
    --vlm_max_tokens 512 \
    --vlm_num_examples 3
```

**Example output:**

```
=================================================================
  AI-GENERATED RADIOLOGY REPORT  (llava-llama3:8b Few-Shot)
  Model: llava-llama3:8b @ http://localhost:11434
=================================================================

FINDINGS:
The cardiac silhouette is mildly enlarged. The mediastinal contours
are within normal limits. There is bilateral hazy opacification in
the lower lung zones, more pronounced on the right, consistent with
pleural effusions. Mild vascular congestion is noted, suggesting
early pulmonary edema. No pneumothorax is identified. No acute
osseous abnormality. An endotracheal tube is present in standard
position.

IMPRESSION:
1. Mild cardiomegaly with pulmonary vascular congestion.
2. Bilateral pleural effusions, right greater than left.
3. Findings suggestive of congestive heart failure.
4. Endotracheal tube in satisfactory position.

-----------------------------------------------------------------
  Generated using 3 similar case captions as context
  Example similarity range: 0.812 - 0.941
  Temperature: 0.3 | Max tokens: 512
=================================================================
```


#### Method 4: Local MedGemma VLM Report (`vlm_few_shot`)

Generates a comprehensive radiology report using **MedGemma 4B-IT** (google/gemma-3-4b-it) running locally. Unlike Ollama which sends only one image, MedGemma natively supports **multi-image inputs** — the model receives the query image **AND** up to 3 retrieved similar images alongside their captions, enabling true visual few-shot learning.

**How it works:**
1. Retrieve top-K similar cases, their captions, and their images from the FAISS index
2. Detect conditions from retrieved captions (used as supplementary context)
3. Build a multi-image chat prompt with:
   - System prompt (expert radiologist persona)
   - Detected conditions summary from database analysis
   - Retrieved similar images (1–3) paired with their captions and similarity scores
   - The query image
   - Output format instructions (FINDINGS + IMPRESSION)
4. Pass all images + prompt through MedGemma's processor and chat template
5. MedGemma generates a detailed medical report by visually comparing the query image against the retrieved examples
6. Clean and format the output

---

### First-Time Setup: Download the Model

MedGemma must be downloaded once before use. The model is cached locally in `./cache/huggingface` and reused for all subsequent runs.

**Step 1: Set your HuggingFace token**

```bash
# Option A: Environment variable (recommended)
export HF_TOKEN="hf_your_token_here"

# Option B: Login via CLI (one time)
huggingface-cli login

# Option C: Set directly in config.py
# In VLMConfig: hf_token = "hf_your_token_here"
```

> **Note:** `google/gemma-3-4b-it` is an open model and does not require license acceptance. If you switch to `google/medgemma-4b-it`, you must first accept the license at [huggingface.co/google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it).

**Step 2: Download the model**

```bash
# Via the pipeline CLI (recommended)
python main_pipeline.py --mode download_model
```

### Usage

**Few-shot generation** (recommended — sends query image + 1–3 similar images + captions):
```bash
python main_pipeline.py --mode query_vlm \
    --query_image /path/to/chest_xray.jpg \
    --checkpoint final_model \
    --top_k 3 \
    --vlm_backend local
```
---

### Example Output

```
=================================================================
  AI-GENERATED RADIOLOGY REPORT
  Model: MedGemma 4B-IT (google/gemma-3-4b-it)
  Mode:  Multi-Image Few-Shot | Backend: local
=================================================================

FINDINGS:
The cardiac silhouette is mildly enlarged, consistent with
cardiomegaly. The mediastinal contours are unremarkable. There is
bilateral hazy opacification in the lower lung zones, right greater
than left, consistent with pleural effusions. Mild pulmonary
vascular congestion is noted with cephalization of the vessels,
suggesting interstitial edema. The lungs are otherwise clear
without focal consolidation or mass. No pneumothorax. An
endotracheal tube is present with the tip approximately 4 cm above
the carina, in satisfactory position. A central venous catheter is
seen with the tip in the superior vena cava.

IMPRESSION:
1. Mild cardiomegaly with pulmonary vascular congestion, findings
   consistent with congestive heart failure.
2. Bilateral pleural effusions, right greater than left.
3. Endotracheal tube and central venous catheter in appropriate
   position.
4. No pneumothorax or focal consolidation.

-----------------------------------------------------------------
  Generated using 3 similar cases as context (4 images sent to model)
  Similarity score range: 0.812 – 0.941
  Temperature: 0.3 | Max tokens: 1024
=================================================================
```

---

## ⚠️ Disclaimer

This system is a **research prototype** for medical image retrieval and report generation. It is **not** a certified medical device and should **not** be used for clinical diagnosis. All generated reports require review by a qualified radiologist.

## Authors
This project was developed by:

Nadeesha Perera
Alli Raittinen

If you use this work in research or products, please cite this repository
and acknowledge the authors.


## License
