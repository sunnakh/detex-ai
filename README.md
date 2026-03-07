# detex.ai — High-Accuracy Local AI Text Detector

> Detect AI-generated text in ~100ms, entirely on your own hardware. No external APIs. No data leaves your machine.

**detex.ai** is a full-stack AI text detection platform built from the ground up. At its core is a custom **LoRA fine-tuned** version of `jinaai/jina-embeddings-v5-text-small` trained on ~350K human/AI text pairs. The resulting embeddings power a calibrated classifier that achieves state-of-the-art detection accuracy with minimal latency.

---

## ✨ Features

- 🔍 **~100ms inference** — local FastAPI + PyTorch backend, no API round-trips
- 🧠 **Fine-tuned Jina v5 embeddings** — LoRA-adapted on HC3, MAGE, RAID, and AI-Detection Pile datasets
- 🎯 **Calibrated classifier** — threshold-tuned Logistic Regression / XGBoost / LightGBM pipeline
- 📄 **File upload support** — detect AI text in `.txt`, `.pdf`, and `.docx` files (up to 10 MB)
- 🔐 **Supabase authentication** — email/password + optional Google OAuth
- 💾 **Persistent chat history** — sessions stored per-user in Supabase, synced across devices
- 🐳 **Docker-first** — entire stack (frontend + backend) launches with a single command
- ⚡ **Apple Silicon support** — MPS acceleration on M1/M2/M3 Macs; CUDA on NVIDIA GPUs

---

## 🏗️ Architecture

```
detex-ai/
├── frontend/               ← Next.js 16 app (App Router, TypeScript, Tailwind CSS 4)
├── backend/                ← FastAPI inference server
│   └── app/main.py         ← /detect + /detect/file endpoints + model lifecycle
├── ml/
│   ├── fine-tune/          ← LoRA fine-tuning pipeline (Stage 1)
│   └── train/              ← Classifier training + selection (Stage 2)
├── checkpoints/            ← Saved LoRA adapter weights
├── docker-compose.yml      ← Full-stack deployment
└── requirements.txt        ← Python dependencies
```

### Inference Pipeline

```
Text Input (or File Upload)
    │
    ▼
Jina v5 (fine-tuned) Tokenizer   ← max_length=256, truncation
    │
    ▼
Transformer → Mean Pool → L2 Normalize   ← 768-dim embedding
    │
    ▼
Calibrated Classifier (best_clf.joblib)
    │
    ▼
{ label, confidence, ai_score, human_score, analysis_time_ms }
```

---

## ⚡ Tech Stack

### Frontend
| Technology | Version | Purpose |
|---|---|---|
| Next.js | 16.1.6 | App framework (App Router) |
| React | 19.2.3 | UI library |
| TypeScript | 5.x | Type safety |
| Tailwind CSS | 4.x | Styling |
| Supabase JS | 2.x | Auth + database client |

### Backend
| Technology | Version | Purpose |
|---|---|---|
| FastAPI | 0.110+ | REST API server |
| PyTorch | 2.0+ | Tensor ops + device management |
| Hugging Face Transformers | 4.40+ | Jina v5 model loading |
| scikit-learn | 1.8.0 | Logistic Regression |
| XGBoost | 2.0+ | Gradient boosting classifier |
| LightGBM | 4.3+ | Gradient boosting (alternative) |
| joblib | 1.3+ | Model serialization |
| pdfplumber | — | PDF text extraction |
| python-docx | — | DOCX text extraction |

### ML Training
| Technology | Purpose |
|---|---|
| sentence-transformers | Fine-tuning and embedding generation |
| PEFT / LoRA | Parameter-efficient fine-tuning |
| Matryoshka Loss + MNRL | Training objective |
| Datasets (HF) | HC3, MAGE, RAID, AI-Detection Pile |

---

## 🚀 Quick Start

### Option A — Docker (Recommended)

**1. Clone and configure environment variables**

```bash
git clone https://github.com/sunnakh/detex-ai.git
cd detex-ai
```

Create a `.env` file at the project root:
```env
NEXT_PUBLIC_SUPABASE_URL=your-supabase-project-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_SITE_URL=http://localhost:3000
```

**2. Build and launch**

```bash
# First-time build (downloads model + installs deps — takes ~10 min)
docker compose build --no-cache && docker compose up -d

# All subsequent starts (instant)
docker compose up -d
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

> **Note:** The Jina embedding model (~500MB) is baked into the Docker image at build time. After the first build, startup is instant — no download delays.

---

### Option B — Local Development

#### Prerequisites

| Requirement | Minimum Version |
|---|---|
| Node.js | 18+ |
| Python | 3.10+ |
| Supabase account | — |

#### 1. Supabase Setup

Create a free project at [supabase.com](https://supabase.com), then run this migration in the **SQL Editor**:

```sql
create table sessions (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid references auth.users(id) on delete cascade not null,
  title      text not null default 'New analysis',
  created_at timestamptz default now()
);

create table messages (
  id          uuid primary key default gen_random_uuid(),
  session_id  uuid references sessions(id) on delete cascade not null,
  role        text not null check (role in ('user','result','error')),
  text        text,
  result      jsonb,
  created_at  timestamptz default now()
);

alter table sessions enable row level security;
alter table messages enable row level security;

create policy "users own sessions" on sessions for all using (user_id = auth.uid());
create policy "users own messages" on messages for all
  using (session_id in (select id from sessions where user_id = auth.uid()));
```

Enable **Email** authentication in your Supabase Auth Providers. Optionally enable **Google OAuth**.  
Add `http://localhost:3000/auth/callback` as a valid Auth Redirect URI.

#### 2. Start the Backend

```bash
# From project root
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at `http://localhost:8000`.

#### 3. Start the Frontend

```bash
cd frontend

# Create environment file
cat > .env.local << 'EOF'
NEXT_PUBLIC_SUPABASE_URL=your-supabase-project-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_API_URL=http://localhost:8000
EOF

npm install
npm run dev
```

The app is available at `http://localhost:3000`.

---

## 🌐 Using the App

1. Navigate to **http://localhost:3000**
2. Sign up with email/password *(or Google if configured)*
3. Paste any text into the chat interface — or upload a `.txt`, `.pdf`, or `.docx` file
4. The detector returns a verdict in ~100ms with confidence scores
5. All sessions are saved automatically and persist across devices

### API Examples (direct)

**Text detection:**
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog."}'
```

**File detection:**
```bash
curl -X POST http://localhost:8000/detect/file \
  -F "file=@/path/to/document.pdf"
```

**Response:**
```json
{
  "label": "Human-written",
  "confidence": 0.8721,
  "ai_score": 0.1279,
  "human_score": 0.8721,
  "word_count": 9,
  "char_count": 44,
  "analysis_time_ms": 97.43
}
```

---

## 🧠 ML Pipeline

The classifier was trained in a two-stage pipeline. See [`ml/README.md`](ml/README.md) for full details.

### Stage 1 — LoRA Fine-Tuning (`ml/fine-tune/`)

A LoRA adapter is trained on top of `jinaai/jina-embeddings-v5-text-small` to specialize the embedding space for human-vs-AI discrimination.

**Training datasets (~350K triplets total):**

| Dataset | Cap | Description |
|---|---|---|
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | 100K pairs | Human vs. ChatGPT answers |
| [MAGE](https://huggingface.co/datasets/yaful/MAGE) | 100K pairs | Multi-domain, multi-generator |
| [RAID](https://huggingface.co/datasets/liamdugan/raid) | 50K pairs | Robustness adversarial set |
| [AI Detection Pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile) | 100K pairs | Diverse human/AI pile |

**Hard negative mining** is applied to augment the dataset with the most confusing AI samples (30% of final dataset).

**LoRA configuration:**

| Hyperparameter | Value |
|---|---|
| Base model | `jinaai/jina-embeddings-v5-text-small` |
| LoRA rank (`r`) | 32 |
| LoRA alpha | 64 |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| Dropout | 0.05 |
| Loss | `MatryoshkaLoss` (dims: 1024, 512, 256) + MNRL |
| Epochs | 1 |
| Effective batch size | 32 |
| Learning rate | 2e-4 |
| Max sequence length | 256 |

**Run fine-tuning:**
```bash
cd ml/fine-tune
python data_pipeline.py   # Build and cache the dataset
python train.py           # Fine-tune (requires 16GB+ VRAM)
```

---

### Stage 2 — Classifier Training (`ml/train/`)

The fine-tuned embedding model generates 768-dim embeddings for the labeled corpus. Three classifiers are trained, evaluated, and the best is selected.

**Classifiers trained:**

| Model | Details |
|---|---|
| Logistic Regression | `C=1.0`, balanced class weight |
| XGBoost | `n_estimators=300`, `max_depth=6`, `lr=0.05` |
| LightGBM | Same hyperparameters as XGBoost |

The pipeline also includes **threshold tuning** (to optimize recall/precision trade-off) and **probability calibration** (Platt scaling).

**Run classifier training (in order):**
```bash
cd ml/train
python data_train.py            # 1. Download + embed corpus
python train_classifiers.py     # 2. Train all classifiers
python evaluating.py            # 3. Evaluate on held-out test set
python calibration_ensemble.py  # 4. Probability calibration (Platt)
python select_model.py          # 5. Select best model by ROC-AUC
python threshold_tuning.py      # 6. Tune decision threshold
```

Copy the selected model to the backend:
```bash
cp ml/train/artifacts/models/<best_model>.joblib backend/artifacts/jina-v5-finetuned/best_clf.joblib
```

---

## 📁 Project Structure

```
detex-ai/
├── backend/
│   ├── Dockerfile
│   ├── artifacts/
│   │   └── jina-v5-finetuned/
│   │       └── best_clf.joblib     ← Production classifier
│   └── app/
│       └── main.py                 ← FastAPI app + /detect + /detect/file endpoints
│
├── frontend/
│   ├── Dockerfile
│   ├── src/                        ← Next.js App Router pages + components
│   └── public/                     ← Static assets
│
├── ml/
│   ├── fine-tune/
│   │   ├── config.py               ← Hyperparameters (single source of truth)
│   │   ├── data_pipeline.py        ← Dataset building + hard negative mining
│   │   ├── train.py                ← LoRA training loop
│   │   ├── inference.py            ← CLI inference tool
│   │   └── model.py                ← Model architecture helpers
│   └── train/
│       ├── config.py
│       ├── data_train.py           ← Corpus download + embedding generation
│       ├── train_classifiers.py    ← Train LogReg / XGBoost / LightGBM
│       ├── evaluating.py           ← Metrics on held-out set
│       ├── calibration_ensemble.py ← Platt scaling
│       ├── select_model.py         ← Best model selection by ROC-AUC
│       └── threshold_tuning.py     ← Decision threshold optimization
│
├── checkpoints/                    ← LoRA adapter weights (git-ignored, large)
├── docker-compose.yml              ← Full-stack deployment
├── requirements.txt                ← Python dependencies
└── pyproject.toml
```

---

## 🔧 Environment Variables Reference

### Frontend (`frontend/.env.local`)

| Variable | Required | Description |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | ✅ | Your Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | ✅ | Supabase anonymous key |
| `NEXT_PUBLIC_API_URL` | ✅ | Backend API URL (e.g. `http://localhost:8000`) |

### Root `.env` (Docker Compose)

| Variable | Required | Description |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | ✅ | Your Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | ✅ | Supabase anonymous key |
| `NEXT_PUBLIC_SITE_URL` | ✅ | Public site URL (e.g. `http://localhost:3000`) |

---

## 🐳 Docker Reference

```bash
# First-time build and start (bakes model into image — ~10 min)
docker compose build --no-cache && docker compose up -d

# Subsequent starts (instant)
docker compose up -d

# View logs
docker compose logs -f

# Stop everything
docker compose down

# Rebuild backend only (e.g. after swapping best_clf.joblib)
docker compose build backend && docker rm -f detex-backend && docker compose up -d

# Full reset (removes all images and volumes)
docker compose down && docker rmi detex-ai-backend detex-ai-frontend
```

**Services:**

| Service | Container | Port |
|---|---|---|
| Backend (FastAPI) | `detex-backend` | 8000 |
| Frontend (Next.js) | `detex-frontend` | 3000 |

The frontend container waits for the backend `/health` endpoint to return `{"model_ready": true}` before starting.

---

## 🖥️ Backend API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns model readiness status |
| `/detect` | POST | Run detection on submitted text |
| `/detect/file` | POST | Run detection on uploaded file (`.txt`, `.pdf`, `.docx`) |
| `/docs` | GET | Interactive Swagger UI |

**`POST /detect` — Request body:**

```json
{
  "text": "string (required)",
  "session_id": "string (optional)"
}
```

**`POST /detect/file` — Request:**

Multipart form with a single `file` field. Max size: 10 MB. Accepted types: `.txt`, `.pdf`, `.docx`.

**Response (both endpoints):**

```json
{
  "label": "AI-generated | Human-written",
  "confidence": 0.9213,
  "ai_score": 0.9213,
  "human_score": 0.0787,
  "word_count": 120,
  "char_count": 714,
  "analysis_time_ms": 98.5
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: your feature"`
4. Push to your fork: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is provided for educational and research purposes.


---

## 🏗️ Architecture

```
detex-ai/
├── frontend/               ← Next.js 16 app (App Router, TypeScript, Tailwind CSS 4)
├── backend/                ← FastAPI inference server
│   └── app/main.py         ← /detect endpoint + model lifecycle
├── ml/
│   ├── fine-tune/          ← LoRA fine-tuning pipeline (Step 1)
│   └── train/              ← Classifier training + selection (Step 2)
├── checkpoints/            ← Saved LoRA adapter weights
├── docker-compose.yml      ← Full-stack deployment
└── requirements.txt        ← Python dependencies
```

### Inference Pipeline

```
Text Input
    │
    ▼
Jina v5 (fine-tuned) Tokenizer   ← max_length=256, truncation
    │
    ▼
Transformer → Mean Pool → L2 Normalize   ← 768-dim embedding
    │
    ▼
Calibrated Classifier (best_clf.joblib)
    │
    ▼
{ label, confidence, ai_score, human_score, analysis_time_ms }
```

---

## ⚡ Tech Stack

### Frontend
| Technology | Version | Purpose |
|---|---|---|
| Next.js | 16.1.6 | App framework (App Router) |
| React | 19.2.3 | UI library |
| TypeScript | 5.x | Type safety |
| Tailwind CSS | 4.x | Styling |
| Supabase JS | 2.x | Auth + database client |

### Backend
| Technology | Version | Purpose |
|---|---|---|
| FastAPI | 0.110+ | REST API server |
| PyTorch | 2.0+ | Tensor ops + device management |
| Hugging Face Transformers | 4.40+ | Jina v5 model loading |
| scikit-learn | 1.4+ | SVM, Logistic Regression |
| XGBoost | 2.0+ | Gradient boosting classifier |
| LightGBM | 4.3+ | Gradient boosting (alternative) |
| joblib | 1.3+ | Model serialization |

### ML Training
| Technology | Purpose |
|---|---|
| sentence-transformers | Fine-tuning and embedding generation |
| PEFT / LoRA | Parameter-efficient fine-tuning |
| Matryoshka Loss + MNRL | Training objective |
| Datasets (HF) | HC3, MAGE, RAID, AI-Detection Pile |

---

## 🚀 Quick Start

### Option A — Docker (Recommended)

This runs the full frontend + backend stack with a single command.

**1. Clone and configure environment variables**

```bash
git clone <repo-url>
cd detex-ai
```

Create a `.env` file at the project root:
```env
NEXT_PUBLIC_SUPABASE_URL=your-supabase-project-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_SITE_URL=http://localhost:3000
```

**2. Launch the stack**

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

> **Note:** On first boot, the backend will download `jinaai/jina-embeddings-v5-text-small` (~500MB) from Hugging Face. Subsequent starts load instantly from the Docker volume cache.

---

### Option B — Local Development

#### Prerequisites

| Requirement | Minimum Version |
|---|---|
| Node.js | 18+ |
| Python | 3.10+ |
| Supabase account | — |

#### 1. Supabase Setup

Create a free project at [supabase.com](https://supabase.com), then run this migration in the **SQL Editor**:

```sql
create table sessions (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid references auth.users(id) on delete cascade not null,
  title      text not null default 'New analysis',
  created_at timestamptz default now()
);

create table messages (
  id          uuid primary key default gen_random_uuid(),
  session_id  uuid references sessions(id) on delete cascade not null,
  role        text not null check (role in ('user','result','error')),
  text        text,
  result      jsonb,
  created_at  timestamptz default now()
);

alter table sessions enable row level security;
alter table messages enable row level security;

create policy "users own sessions" on sessions for all using (user_id = auth.uid());
create policy "users own messages" on messages for all
  using (session_id in (select id from sessions where user_id = auth.uid()));
```

Enable **Email** authentication in your Supabase Auth Providers. Optionally enable **Google OAuth**.  
Add `http://localhost:3000/auth/callback` as a valid Auth Redirect URI.

#### 2. Start the Backend

```bash
# From project root
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at `http://localhost:8000`. On first launch, it downloads the Jina v5 model weights.

#### 3. Start the Frontend

```bash
# From project root
cd frontend

# Create environment file
cat > .env.local << 'EOF'
NEXT_PUBLIC_SUPABASE_URL=your-supabase-project-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_API_URL=http://localhost:8000
EOF

npm install
npm run dev
```

The app is available at `http://localhost:3000`.

---

## 🌐 Using the App

1. Navigate to **http://localhost:3000**
2. Sign up with email/password *(or Google if configured)*
3. Paste any text into the chat interface
4. The detector returns a verdict in ~100ms with confidence scores
5. All sessions are saved automatically and persist across devices

### API Example (direct)

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog."}'
```

**Response:**
```json
{
  "label": "Human-written",
  "confidence": 0.8721,
  "ai_score": 0.1279,
  "human_score": 0.8721,
  "word_count": 9,
  "char_count": 44,
  "analysis_time_ms": 97.43
}
```

---

## 🧠 ML Pipeline

The classifier was trained in a two-stage pipeline. See [`ml/README.md`](ml/README.md) for full details.

### Stage 1 — LoRA Fine-Tuning (`ml/fine-tune/`)

A LoRA adapter is trained on top of `jinaai/jina-embeddings-v5-text-small` to specialize the embedding space for human-vs-AI discrimination.

**Training datasets (~350K triplets total):**

| Dataset | Cap | Description |
|---|---|---|
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | 100K pairs | Human vs. ChatGPT answers |
| [MAGE](https://huggingface.co/datasets/yaful/MAGE) | 100K pairs | Multi-domain, multi-generator |
| [RAID](https://huggingface.co/datasets/liamdugan/raid) | 50K pairs | Robustness adversarial set |
| [AI Detection Pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile) | 100K pairs | Diverse human/AI pile |

**Hard negative mining** is applied to augment the dataset with the most confusing AI samples (30% of final dataset).

**LoRA configuration:**

| Hyperparameter | Value |
|---|---|
| Base model | `jinaai/jina-embeddings-v5-text-small` |
| LoRA rank (`r`) | 32 |
| LoRA alpha | 64 |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| Dropout | 0.05 |
| Loss | `MatryoshkaLoss` (dims: 1024, 512, 256) + MNRL |
| Epochs | 1 (RTX 5090 target) |
| Effective batch size | 32 |
| Learning rate | 2e-4 |
| Max sequence length | 256 |

**Run fine-tuning:**
```bash
cd ml/fine-tune
python data_pipeline.py   # Build and cache the dataset
python train.py           # Fine-tune (requires 16GB+ VRAM)
```

---

### Stage 2 — Classifier Training (`ml/train/`)

The fine-tuned embedding model is used to generate 768-dim embeddings for the labeled corpus. Multiple classifiers are trained, evaluated, and the best is selected.

**Classifiers trained:**

| Model | Details |
|---|---|
| Logistic Regression | `C=1.0`, balanced class weight |
| XGBoost | `n_estimators=300`, `max_depth=6`, `lr=0.05` |
| LightGBM | Same hyperparameters as XGBoost |

The pipeline also includes **threshold tuning** (to optimize recall/precision trade-off) and **probability calibration** (Platt scaling).

**Run classifier training:**
```bash
cd ml/train
python data_train.py            # Download + split corpus
python train_classifiers.py     # Train all classifiers
python evaluating.py            # Evaluate on held-out test set
python select_model.py          # Select best model by F1 score
python threshold_tuning.py      # Tune decision threshold
python calibration_ensemble.py  # Probability calibration (Platt)
```

**Or run everything via Docker (GPU required):**
```bash
docker compose -f ml/docker-compose.yml up --build
```

---

## 📁 Project Structure

```
detex-ai/
├── backend/
│   ├── Dockerfile
│   ├── artifacts/
│   │   └── jina-v5-finetuned/
│   │       └── best_clf.joblib     ← Production classifier
│   └── app/
│       └── main.py                 ← FastAPI app + /detect endpoint
│
├── frontend/
│   ├── Dockerfile
│   ├── src/                        ← Next.js App Router pages + components
│   └── public/                     ← Static assets
│
├── ml/
│   ├── fine-tune/
│   │   ├── config.py               ← Hyperparameters (single source of truth)
│   │   ├── data_pipeline.py        ← Dataset building + hard negative mining
│   │   ├── train.py                ← LoRA training loop
│   │   ├── inference.py            ← CLI inference tool
│   │   └── model.py                ← Model architecture helpers
│   └── train/
│       ├── config.py
│       ├── data_train.py           ← Corpus download + embedding generation
│       ├── train_classifiers.py    ← Train LogReg / XGBoost / LightGBM
│       ├── evaluating.py           ← Metrics on held-out set
│       ├── select_model.py         ← Best model selection by F1
│       ├── threshold_tuning.py     ← Decision threshold optimization
│       └── calibration_ensemble.py ← Platt scaling
│
├── checkpoints/                    ← LoRA adapter weights (git-ignored, large)
├── docker-compose.yml              ← Full-stack deployment
├── requirements.txt                ← Python dependencies
└── pyproject.toml
```

---

## 🔧 Environment Variables Reference

### Frontend (`frontend/.env.local`)

| Variable | Required | Description |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | ✅ | Your Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | ✅ | Supabase anonymous key |
| `NEXT_PUBLIC_API_URL` | ✅ | Backend API URL (e.g. `http://localhost:8000`) |

### Root `.env` (Docker Compose)

| Variable | Required | Description |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | ✅ | Your Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | ✅ | Supabase anonymous key |
| `NEXT_PUBLIC_SITE_URL` | ✅ | Public site URL (e.g. `http://localhost:3000`) |

---

## 🐳 Docker Reference

```bash
# Build and start full stack
docker compose up --build

# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop everything
docker compose down

# Stop and remove volumes (clears HF cache)
docker compose down -v
```

**Services:**

| Service | Container | Port |
|---|---|---|
| Backend (FastAPI) | `detex-backend` | 8000 |
| Frontend (Next.js) | `detex-frontend` | 3000 |

The frontend container waits for the backend `/health` endpoint to return `{"model_ready": true}` before starting.

---

## 🖥️ Backend API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns model readiness status |
| `/detect` | POST | Run detection on submitted text |
| `/docs` | GET | Interactive Swagger UI |

**`POST /detect` — Request body:**

```json
{
  "text": "string (required)",
  "session_id": "string (optional)"
}
```

**`POST /detect` — Response:**

```json
{
  "label": "AI-generated | Human-written",
  "confidence": 0.9213,
  "ai_score": 0.9213,
  "human_score": 0.0787,
  "word_count": 120,
  "char_count": 714,
  "analysis_time_ms": 98.5
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and commit: `git commit -m "feat: your feature"`
4. Push to your fork: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is provided for educational and research purposes.
