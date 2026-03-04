# ML Pipeline

This folder contains the full ML lifecycle for the **Detex AI** text detection system.
There are three sub-modules that run in sequence to produce a production-ready detector.

```
ml/
├── fine-tune/          ← Step 1 · LoRA fine-tune Jina v5 on AI-detection task
├── train/              ← Step 2 · Embed corpus → train classifiers → select best
├── pre-trained/        ← Standalone detector using the base model (no training needed)
└── docker-compose.yml  ← Runs Step 1 + Step 2 back-to-back with GPU support
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| CUDA (GPU training) | 12.1+ |
| Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | Any recent |

Install Python dependencies (for local runs):
```bash
pip install -r requirements.txt   # project root requirements.txt
```

---

## Option A — Full Pipeline via Docker (Recommended)

Run both fine-tuning and classifier training in one command from the **project root**:

```bash
# Build both images
docker compose -f ml/docker-compose.yml build

# Run: fine-tune → then classifier pipeline (GPU required)
docker compose -f ml/docker-compose.yml up
```

Both containers are one-shot jobs — they exit once done. All outputs are saved to named Docker volumes and also persist on your host machine under `checkpoints/` and `ml/train/artifacts/`.

To run fully from scratch (clears all volumes):
```bash
docker compose -f ml/docker-compose.yml down -v
docker compose -f ml/docker-compose.yml up
```

---

## Option B — Run Locally (Step by Step)

### Step 1 · Fine-Tune Jina v5 (`fine-tune/`)

Produces a LoRA adapter saved to `checkpoints/jina-v5-ai-detection-final/`.

```bash
cd ml/fine-tune

# 1a. Build the dataset (downloads HC3, MAGE, RAID, Pile datasets)
python data_pipeline.py

# 1b. Fine-tune the model (requires a GPU with 16GB+ VRAM)
python train.py
```

**Key config values** (`fine-tune/config.py`):

| Setting | Default | Description |
|---|---|---|
| `MODEL_ID` | `jinaai/jina-embeddings-v5-text-small` | Base model |
| `EPOCHS` | `1` | Training epochs |
| `BATCH_SIZE` | `4` | Per-device batch size |
| `GRAD_ACCUM` | `4` | Effective batch = 16 |
| `MAX_SEQ_LENGTH` | `256` | Truncate input to this |
| `LORA_R` | `32` | LoRA rank |
| `CHECKPOINT_DIR` | `checkpoints/jina-v5-ai-detection` | Mid-training saves |
| `FINAL_DIR` | `checkpoints/jina-v5-ai-detection-final` | Final adapter |

**Output** (saved at project root):
```
checkpoints/
├── jina-v5-ai-detection/           ← intermediate checkpoints
└── jina-v5-ai-detection-final/
    ├── adapter_config.json
    └── adapter_model.safetensors   ← the trained LoRA weights
```

---

### Step 2 · Train Classifiers (`train/`)

Loads the fine-tuned Jina model, generates embeddings, trains and selects the best classifier, and saves a production-ready `.joblib` file.

```bash
cd ml/train

# Runs all stages in order
python data_train.py          # Download + split corpus
python train_classifiers.py   # Train Logistic Regression, SVM, XGBoost, etc.
python evaluating.py          # Evaluate on held-out test set
python select_model.py        # Pick best model by F1
python threshold_tuning.py    # Find optimal decision threshold
python calibration_ensemble.py  # Probability calibration
```

**Output** (saved to `ml/train/artifacts/`):
```
ml/train/artifacts/
├── embeddings_train.npy      ← cached embeddings (avoids re-running Jina)
├── embeddings_test.npy
├── best_classifier.joblib    ← final production classifier
├── model_selection.json      ← F1 / AUC scores for all models
└── optimal_threshold.json    ← tuned decision threshold
```

---

### Step 3 · Run Inference

#### Fine-tuned model (`fine-tune/inference.py`)

Uses the LoRA adapter from Step 1, compared against the same anchor corpus as `detect.py` for fair comparison.

```bash
cd ml/fine-tune
python inference.py
```

- Paste or type text, press **Enter twice** to analyze.
- Type `q` or `quit` to exit.

#### Pre-trained model (`pre-trained/detect.py`)

Standalone detector — no training needed. Uses the base Jina v5 model directly.

```bash
cd ml/pre-trained
python detect.py
```

Same UX as `inference.py`. Use both side-by-side to compare base vs. fine-tuned performance.

---

## Output Format (both inference scripts)

```
╔══════════════════════════════════════════════════╗
  Verdict   :  AI-GENERATED
  Confidence:  ●●●  (high)
╠══════════════════════════════════════════════════╣
  AI score     73.12%  │██████████████████████    │
  Human score  26.88%  │████████                  │
  Margin       46.24%
╠══════════════════════════════════════════════════╣
  Words: 120   Chars: 714   Time: 231.4 ms
  RAM Usage: 2.34 GB
╚══════════════════════════════════════════════════╝
```

| Field | Meaning |
|---|---|
| **Verdict** | `AI-GENERATED`, `HUMAN-WRITTEN`, or `UNCERTAIN` |
| **Confidence** | `high` (>20% margin), `medium` (5–20%), `low` (<5%) |
| **AI / Human score** | Normalized percentages that always sum to 100% |
| **Margin** | Absolute difference between the two scores |
| **RAM Usage** | Current process memory footprint in GB |

---

## Running Fine-Tuning via Docker (standalone)

If you only want to re-run the fine-tuning step without the classifier pipeline:

```bash
# Build image (from project root)
docker build -f ml/fine-tune/Dockerfile -t detex-finetune .

# Run full pipeline (dataset generation + training)
docker run --gpus all -it --rm \
  -v $(pwd)/ml/fine-tune/.data:/app/ml/fine-tune/.data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v ~/.cache/huggingface:/app/.cache \
  detex-finetune bash -c "python data_pipeline.py && python train.py"
```

The `--gpus all` flag requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to be installed.

---

## Next Steps / Extending

- **Add clustering**: Use the fine-tuned embeddings with `sklearn.cluster.KMeans` or `hdbscan` to discover natural topic clusters in the writing styles.
- **Improve classifier accuracy**: Add more labeled training data in `data_pipeline.py` (increase `HC3_CAP`, `MAGE_CAP`, etc. in `config.py`).
- **Merge LoRA weights**: For simpler deployment, merge the LoRA adapter into the base model to produce a single model file with no PEFT dependency.
