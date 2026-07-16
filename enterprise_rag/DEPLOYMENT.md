# Cloud Deployment Guide
## FastAPI (Render) + Qdrant Cloud + Neo4j Aura + Supabase Storage

---

## Architecture

```
Browser / Frontend (Vercel)
        │
        ▼
FastAPI API (Render Free)
        │
        ├── OpenRouter (LLM)
        ├── Qdrant Cloud (vector DB)
        ├── Neo4j Aura (knowledge graph)
        └── Supabase Storage (file storage)
```

---

## Step 1 — Qdrant Cloud Setup

1. Go to **https://cloud.qdrant.io**
2. Sign up / log in
3. Click **"Create Cluster"**
   - Name: `enterprise-rag`
   - Plan: **Free** (1 GB, 1 node)
   - Region: `us-east-1` (or closest to your Render region)
4. Click **"Create"** — wait ~2 minutes
5. Go to **"API Keys"** tab → click **"Create API Key"**
   - Name: `render-production`
   - Copy the key immediately (shown once)
6. Note your cluster URL from the dashboard:
   ```
   https://xxxxxxxxxxxx.us-east-1-0.aws.cloud.qdrant.io
   ```
7. Set environment variables:
   ```
   QDRANT_URL=https://xxxxxxxxxxxx.us-east-1-0.aws.cloud.qdrant.io
   QDRANT_API_KEY=your-api-key
   QDRANT_COLLECTION=documents
   ```
8. Create the collection (run once, from your local machine):
   ```python
   from qdrant_client import QdrantClient
   from qdrant_client.models import Distance, VectorParams

   client = QdrantClient(
       url="https://xxxxxxxxxxxx.us-east-1-0.aws.cloud.qdrant.io",
       api_key="your-api-key",
   )
   client.create_collection(
       collection_name="documents",
       vectors_config=VectorParams(size=384, distance=Distance.COSINE),
   )
   print("Collection created")
   ```

---

## Step 2 — Neo4j Aura Setup

1. Go to **https://console.neo4j.io**
2. Sign up / log in
3. Click **"New Instance"**
   - Choose **AuraDB Free** (free forever, 200k nodes)
   - Name: `enterprise-rag`
   - Region: closest to Render
4. Click **"Create"** — takes 2–3 minutes
5. Download the connection credentials file shown on screen
6. Your connection details:
   ```
   NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=<generated-password-from-credentials-file>
   NEO4J_DATABASE=neo4j
   ```
   > ⚠️ The password is only shown once — save it from the credentials file.

7. Test connection:
   ```python
   from neo4j import GraphDatabase
   driver = GraphDatabase.driver(
       "neo4j+s://xxxxxxxx.databases.neo4j.io",
       auth=("neo4j", "your-password"),
   )
   driver.verify_connectivity()
   print("Neo4j Aura connected")
   driver.close()
   ```

---

## Step 3 — Supabase Storage Setup

1. Go to **https://supabase.com**
2. Sign up / log in → **"New Project"**
   - Name: `enterprise-rag`
   - Database password: (save this, not needed for storage)
   - Region: closest to Render
3. Wait for project to finish provisioning (~1 minute)
4. Go to **Storage** (left sidebar) → **"New Bucket"**
   - Name: `rag-documents`
   - Public bucket: **Yes** (so uploaded files can be fetched by the ingestion pipeline)
   - File size limit: `52428800` (50 MB)
   - Allowed MIME types: `application/pdf, text/plain, image/png, image/jpeg`
5. Go to **Project Settings** → **API**
   - Copy **Project URL** → `SUPABASE_URL`
   - Copy **service_role key** (under "Project API Keys") → `SUPABASE_KEY`
   > Use the `service_role` key (not `anon`). It bypasses RLS for server-side uploads.
6. Set environment variables:
   ```
   SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
   SUPABASE_KEY=eyJhbGci...your-service-role-key
   SUPABASE_BUCKET=rag-documents
   ```

---

## Step 4 — OpenRouter Setup

1. Go to **https://openrouter.ai/keys**
2. Click **"Create Key"**
   - Name: `enterprise-rag-production`
3. Copy the key (starts with `sk-or-v1-`)
4. Set environment variables:
   ```
   OPENAI_API_KEY=sk-or-v1-your-key
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   OPENAI_MODEL=openai/gpt-4o-mini
   ```

**Free/cheap model options:**
| Model | Cost | Notes |
|---|---|---|
| `openai/gpt-4o-mini` | ~$0.15/1M tokens | Best value, recommended |
| `meta-llama/llama-3.1-8b-instruct:free` | Free | Rate limited |
| `mistralai/mistral-7b-instruct:free` | Free | Rate limited |
| `google/gemma-2-9b-it:free` | Free | Rate limited |
| `anthropic/claude-3-haiku` | ~$0.25/1M tokens | High quality |

---

## Step 5 — Render Deployment

### Option A — Deploy via GitHub (recommended)

1. Push your project to GitHub:
   ```bash
   cd D:\enterprise_rag_complete\enterprise_rag
   git init
   git add .
   git commit -m "Initial commit — cloud migration"
   git remote add origin https://github.com/yourusername/enterprise-rag.git
   git push -u origin main
   ```

2. Go to **https://dashboard.render.com**
3. Click **"New"** → **"Web Service"**
4. Connect GitHub → select your repo
5. Configure:
   - **Name**: `enterprise-rag-api`
   - **Runtime**: Docker
   - **Dockerfile path**: `./Dockerfile`
   - **Plan**: Free

6. Add environment variables (click "Add Environment Variable" for each):
   ```
   OPENAI_API_KEY          = sk-or-v1-your-openrouter-key
   OPENAI_BASE_URL         = https://openrouter.ai/api/v1
   OPENAI_MODEL            = openai/gpt-4o-mini
   QDRANT_URL              = https://xxxx.qdrant.io
   QDRANT_API_KEY          = your-qdrant-key
   QDRANT_COLLECTION       = documents
   NEO4J_URI               = neo4j+s://xxxx.databases.neo4j.io
   NEO4J_USER              = neo4j
   NEO4J_PASSWORD          = your-neo4j-password
   SUPABASE_URL            = https://xxxx.supabase.co
   SUPABASE_KEY            = your-service-role-key
   SUPABASE_BUCKET         = rag-documents
   SECURITY_ENABLED        = true
   METRICS_ENABLED         = false
   ENV                     = prod
   ```

7. Click **"Create Web Service"**
8. Wait for build (~5–10 minutes for first deploy)
9. Your API will be live at: `https://enterprise-rag-api.onrender.com`

### Option B — Deploy via render.yaml blueprint

```bash
# Install Render CLI
npm install -g @render-com/cli

# Login
render login

# Deploy using render.yaml in project root
render blueprint launch
```

---

## Step 6 — Verify Deployment

```bash
# Health check
curl https://enterprise-rag-api.onrender.com/api/v2/health

# Expected response:
# {"status":"ok","version":"2.0.0","components":{"security":"ready","neo4j":"connected"}}

# Test a query
curl -X POST https://enterprise-rag-api.onrender.com/api/v2/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'

# Upload a file
curl -X POST https://enterprise-rag-api.onrender.com/api/v2/upload \
  -F "files=@document.pdf"
```

---

## Local Development with Cloud Services

To develop locally while pointing to cloud services:

```bash
cd D:\enterprise_rag_complete\enterprise_rag
venv\Scripts\activate

# Copy .env.example and fill in cloud credentials
copy .env.example .env
code .env

# Run locally (connects to Qdrant Cloud, Neo4j Aura, Supabase)
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## Local Development with Local Services (fully offline)

```bash
cd D:\enterprise_rag_complete\enterprise_rag

# Override cloud URLs to point to local Docker containers
# Edit .env:
#   QDRANT_URL=http://localhost:6333
#   QDRANT_API_KEY=
#   NEO4J_URI=bolt://localhost:7687
#   NEO4J_PASSWORD=devpassword
#   SUPABASE_URL=  (leave empty — uploads will be skipped gracefully)

docker compose up -d     # starts local Qdrant + Neo4j
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

---

## Deployment Checklist

### Qdrant Cloud
- [ ] Cluster created (free tier)
- [ ] API key generated
- [ ] `documents` collection created with `size=384, distance=Cosine`
- [ ] `QDRANT_URL` and `QDRANT_API_KEY` set in Render

### Neo4j Aura
- [ ] AuraDB Free instance created
- [ ] Credentials file saved (password shown once)
- [ ] Connection tested with Python driver
- [ ] `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` set in Render

### Supabase Storage
- [ ] Project created
- [ ] `rag-documents` bucket created (public)
- [ ] File size limit and MIME types configured
- [ ] `service_role` key copied (not `anon` key)
- [ ] `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_BUCKET` set in Render

### OpenRouter
- [ ] API key created
- [ ] Credit added (minimum $5 recommended) or free model selected
- [ ] `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` set in Render

### Render
- [ ] GitHub repo connected
- [ ] All 15+ environment variables set
- [ ] First deploy succeeded
- [ ] Health check returns 200
- [ ] Query endpoint tested
- [ ] Upload endpoint tested

### Code changes verified
- [ ] `requirements.txt` — `chromadb` removed, `qdrant-client` + `supabase` added
- [ ] `app/config.py` — Qdrant/Supabase/OpenRouter vars, ChromaDB removed
- [ ] `app/hybrid/retriever.py` — uses `QdrantVectorStore`
- [ ] `app/services/retrieval/hybrid_retriever.py` — uses `QdrantVectorStore`
- [ ] `app/vector_store/qdrant_store.py` — new Qdrant wrapper
- [ ] `app/storage/supabase_storage.py` — new Supabase wrapper
- [ ] All `AsyncOpenAI()` calls include `base_url=settings.OPENAI_BASE_URL`
- [ ] `docker-compose.yml` — dev only, uses local Qdrant + Neo4j containers
- [ ] `render.yaml` — production blueprint

---

## Render Free Tier Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| Sleeps after 15 min inactivity | First request after sleep takes ~30s | Use UptimeRobot to ping every 10 min |
| 512 MB RAM | Models may OOM | Disable NLI model: set `HALLUCINATION_NLI_MODEL=none` |
| No persistent disk | BM25 index rebuilt on startup | Index is rebuilt from Qdrant on each cold start |
| 750 compute hours/month | Shared across all free services | Use 1 service |

**Recommended upgrade path**: Render Starter ($7/mo) removes sleep + adds always-on health.

---

## Keep-Alive (prevent Render sleep)

Add this free service to prevent cold starts:
1. Go to **https://uptimerobot.com** → Sign up free
2. Add monitor:
   - Type: HTTP(S)
   - URL: `https://enterprise-rag-api.onrender.com/api/v2/health`
   - Interval: 10 minutes
3. Render will stay warm (free tier allows 14 requests/hour for keeps-alive)
