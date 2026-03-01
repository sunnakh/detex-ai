# detex.ai — High-Accuracy Local AI Text Detector

A full-stack, state-of-the-art AI text detector that runs entirely **locally** on your hardware. 
Unlike most detectors that rely on slow, expensive external APIs, detex.ai uses a fast, local **FastAPI + PyTorch** backend to run Subword Pattern Analysis and classical machine learning (e.g., SVM, XGBoost) natively, ensuring complete privacy, zero API costs, and lightning-fast inference times (~100ms).

The frontend is a beautifully designed, responsive **Next.js** application featuring a stunning animated landing page, a ChatGPT-style detector interface, and is integrated with **Supabase** for secure authentication and persistent cross-device chat history.

---

## ⚡ Tech Stack

**Frontend:**
- Next.js 16 (App Router)
- React 19
- Tailwind CSS
- TypeScript
- Supabase (Auth + Persistent Session Storage)

**Backend:**
- FastAPI (Python 3.10+)
- PyTorch (with Apple Silicon MPS acceleration support)
- Hugging Face `transformers` (Local Embeddings)
- `scikit-learn` & `xgboost` (Local Classifier)

---

## 🚀 Getting Started

Follow these steps to get the full application running locally on your machine.

### 1. Prerequisites
- **Node.js**: v18 or higher (for the Next.js frontend)
- **Python**: v3.10+ (for the FastAPI backend)
- **Supabase Project**: A free Supabase project for authentication and database support.

### 2. Supabase Setup
You need a Supabase project to store user accounts and chat history.
1. Create a project at [supabase.com](https://supabase.com/).
2. In the **SQL Editor**, run the following migration script to create the necessary tables and RLS policies:
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
3. Ensure **Email** Authentication is enabled in your Supabase Auth Providers (we recommend turning off "Confirm email" during development for immediate sign-in). You can optionally enable **Google** OAuth as well. Add `http://localhost:3000/auth/callback` as a valid Auth Redirect URI.

### 3. Environment Variables
Create a `.env.local` file inside the `frontend/` directory:
```env
NEXT_PUBLIC_SUPABASE_URL=your-supabase-project-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🏃‍♂️ Running the Services

The application consists of two decoupled services that must run simultaneously.

### Start the Backend (FastAPI + AI Model)
Open a new terminal window:
```bash
# Navigate to the project root
cd ai-text-detector

# Create and activate a Python virtual environment (if you haven't already)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*Note: The first time you run the backend, it will automatically download the Hugging Face `jinaai/jina-embeddings-v5-text-small` model weights to your machine (~500MB). Subsequent boots will load instantly from your local cache.*

### Start the Frontend (Next.js)
Open a second terminal window:
```bash
# Navigate to the frontend directory
cd ai-text-detector/frontend

# Install node modules
npm install

# Start the development server
npm run dev
```

---

## 🌐 Usage
Once both servers are running:
1. Open your browser and navigate to **http://localhost:3000**.
2. Create an account via email/password or Google sign-in.
3. Paste any text into the chat interface to analyze it in real-time.
4. Your analysis sessions are automatically saved to Supabase and will persist across refreshes!

---

## 🐳 Running with Docker (Optional)
If you prefer not to manage Python and Node.js environments manually, you can run the entire stack via Docker Compose.

Make sure you've copied your Supabase credentials into the main `.env` file at the root of the project:
```env
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_SITE_URL=http://localhost:3000
```

Then, run:
```bash
docker-compose up --build
```
The application will be exposed at `http://localhost:3000` and the API at `http://localhost:8000`.

---

## 🧠 Model Architecture details
This project is built using a custom fine-tuned machine learning pipeline (Baseline classification, SVM, Random Forest, or XGBoost) leveraging high-dimensional embeddings. The model maps variable-length texts into 768-dimensional space via raw `transformers` encoding and mean pooling, bypassing traditional heuristic pitfalls to detect the underlying stylistic variance and uniform predictability inherent in Large Language Models (LLMs).
