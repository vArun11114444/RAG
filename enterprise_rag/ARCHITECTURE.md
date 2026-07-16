# Enterprise Agentic RAG — Architecture Guide

## Overview

Production-grade RAG system with 6 layers:

```
User Request
     │
     ▼
┌─────────────────────────────────┐
│  Security Layer  (NEW)          │  ← Blocks injections, masks PII, validates files
└─────────────────────────────────┘
     │ sanitized_query
     ▼
┌─────────────────────────────────┐
│  Phase 4: Query Planner         │  ← Classifies query, picks strategy
└─────────────────────────────────┘
     │ QueryPlan
     ▼
┌─────────────────────────────────┐
│  Phase 1: Hybrid Retrieval      │  ← BM25 + Vector + RRF + Metadata filter
└─────────────────────────────────┘
     │ ranked chunks
     ▼
┌─────────────────────────────────┐
│  Phase 2: Knowledge Graph       │  ← Entity extraction + Neo4j traversal
└─────────────────────────────────┘  (conditional on query type)
     │ enriched context
     ▼
┌─────────────────────────────────┐
│  LLM Answer Generation          │  ← OpenAI with grounded context
└─────────────────────────────────┘
     │ draft answer
     ▼
┌─────────────────────────────────┐
│  Phase 3: Verification          │  ← Grounding + citation + hallucination check
└─────────────────────────────────┘  (conditional on query type)
     │ verified answer
     ▼
┌─────────────────────────────────┐
│  Phase 5: Observability         │  ← LangSmith + Prometheus + Structured logs
└─────────────────────────────────┘  (wraps every phase)
     │
     ▼
 RAGResponse
```

---

## Folder Structure

```
enterprise_rag/
├── app/
│   ├── config.py                       # Central settings (pydantic-settings, UPPERCASE vars)
│   ├── pipeline.py                     # RAGPipelineExecutor — main orchestrator
│   ├── main.py                         # FastAPI app factory + lifespan
│   │
│   ├── models/
│   │   └── schemas.py                  # All Pydantic schemas (single source of truth)
│   │
│   ├── security/                       # ── SECURITY LAYER (added) ──
│   │   ├── exceptions.py               # SecurityException hierarchy
│   │   ├── pii_detector.py             # Presidio + custom IN_AADHAAR/IN_PAN + regex fallback
│   │   ├── data_masker.py              # REPLACE/REDACT/HASH strategies + audit mapping
│   │   ├── prompt_injection.py         # 21 patterns across 7 attack categories
│   │   ├── file_validator.py           # Size + ext + MIME + magic + PDF/img/zip-bomb
│   │   ├── security_pipeline.py        # SecurityPipeline orchestrator
│   │   ├── metrics.py                  # 9 Prometheus metrics (rag_security_* namespace)
│   │   └── __init__.py                 # Clean public API
│   │
│   ├── hybrid/                         # ── PHASE 1 ──
│   │   ├── bm25_retriever.py           # BM25Okapi index, async to_thread offload
│   │   ├── query_expander.py           # LLM query variant generation
│   │   ├── rrf.py                      # Reciprocal Rank Fusion + min-max normalization
│   │   ├── metadata_filter.py          # exact/range/in/nin + ChromaDB $where translator
│   │   └── retriever.py                # HybridRetriever (unified plan= or query= API)
│   │
│   ├── graph/                          # ── PHASE 2 ──
│   │   ├── extractor.py                # EntityRelationExtractor (spaCy + LLM)
│   │   ├── neo4j_client.py             # Async Neo4j driver, upsert, N-hop traversal
│   │   └── pipeline.py                 # GraphPipeline (extract → store → traverse)
│   │
│   ├── verification/                   # ── PHASE 3 ──
│   │   ├── grounding.py                # Source grounding + citation validation
│   │   ├── contradiction.py            # NLI/heuristic contradiction detection
│   │   ├── confidence.py               # Hallucination check + composite confidence score
│   │   └── pipeline.py                 # VerificationPipeline orchestrator
│   │
│   ├── planner/                        # ── PHASE 4 ──
│   │   ├── classifier.py               # Rule-based pattern classifier
│   │   └── planner.py                  # QueryPlanner → QueryPlan (rule + LLM fallback)
│   │
│   ├── observability/                  # ── PHASE 5 ──
│   │   ├── logger.py                   # Structured JSON logger + contextvars + Timer
│   │   ├── metrics.py                  # Prometheus counters/histograms/gauges
│   │   └── tracer.py                   # LangSmith trace_span + @traced decorator
│   │
│   ├── core/
│   │   ├── config.py                   # Alternate settings (lowercase, services layer)
│   │   └── dependencies.py             # FastAPI DI factory (get_pipeline, get_security)
│   │
│   ├── services/                       # Original services implementation
│   │   ├── pipeline.py                 # AgenticRAGPipeline (security-aware)
│   │   ├── retrieval/                  # BM25, hybrid, RRF, query expansion
│   │   ├── knowledge_graph/            # Neo4j store, extractor, graph service
│   │   ├── planner/                    # Query planner
│   │   ├── verification/               # Answer verifier
│   │   └── observability/              # Logger, metrics, tracing
│   │
│   └── api/
│       └── routes/
│           └── query.py                # POST /api/v2/query
│                                       # POST /api/v2/upload    (file upload + security)
│                                       # POST /api/v2/admin/index-bm25
│                                       # GET  /api/v2/health
│
├── tests/
│   ├── conftest.py                     # Shared pytest fixtures + stubs
│   ├── test_security.py                # PII, injection, file, pipeline (154 tests)
│   ├── test_fusion.py                  # RRF + normalization unit tests
│   ├── test_planner.py                 # Query classifier unit tests
│   ├── test_verification.py            # Verification layer unit tests
│   └── test_pipeline.py                # Integration tests (mocked deps)
│
├── config/
│   ├── prometheus.yml                  # Prometheus scrape config
│   ├── alerts.yml                      # Alert rules (security + quality + availability)
│   └── grafana/
│       └── provisioning/
│           ├── datasources/            # Auto-connect Prometheus datasource
│           └── dashboards/             # Auto-load RAG dashboard
│
├── docker-compose.yml                  # API + ChromaDB + Neo4j + Prometheus + Grafana
├── Dockerfile                          # Multi-stage, non-root, libmagic + spaCy model
├── Makefile                            # Developer convenience commands
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Security Layer Details

### Detection Pipeline
```
SecurityRequest { query, uploaded_files }
        │
        ▼ Step 1
File Validation (each file)
  ├── Size limit (default 50 MB)
  ├── Extension allowlist (.pdf, .txt, .csv, .png, .jpg, .docx, .xlsx …)
  ├── Declared MIME type allowlist
  ├── Magic bytes verification (actual content vs declared type)
  ├── Malicious signature scan (EICAR, PE header, ELF, shell shebang, JS protocol)
  ├── PDF-specific: /JavaScript, /Launch, /OpenAction detection
  ├── Image-specific: PIL validation, decompression bomb check
  └── ZIP bomb: ratio + uncompressed size limits
        │
        ▼ Step 2
Prompt Injection Detection (synchronous regex, < 1ms)
  ├── 21 patterns across 7 categories
  ├── Instruction override: "ignore previous instructions", "disregard all rules"
  ├── System extraction: "reveal system prompt", "print hidden instructions"
  ├── Jailbreak: DAN, developer mode, no-restrictions roleplay
  ├── Role hijack: "you are now an evil AI"
  ├── Data exfiltration: "dump all documents", "list entire knowledge base"
  ├── Encoding attacks: base64 decode + RTLO unicode detection
  └── Delimiter injection: <|im_end|>, [INST], ###System: boundary probing
        │
        ▼ Step 3
PII Detection (async, Presidio or regex fallback)
  ├── EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, IP_ADDRESS
  ├── IN_AADHAAR (custom recognizer — 12-digit, UIDAI spec)
  ├── IN_PAN (custom recognizer — AAAAA9999A format)
  └── PERSON, IBAN_CODE, URL (via Presidio spaCy NLP engine)
        │
        ▼ Step 4 (if PII found)
Data Masking
  ├── REPLACE  → <EMAIL_ADDRESS>   (default for emails, phones, IPs)
  ├── HASH     → [HASH:ab3f]       (default for financial IDs: CC, AADHAAR, PAN)
  └── REDACT   → [REDACTED]        (default fallback)
        │
        ▼
SecurityResult {
  sanitized_query,        ← PII-masked; used by all downstream phases
  security_score,         ← 0=dangerous, 1=clean
  injection_risk,         ← 0.0–1.0
  pii_entity_count,
  blocked,                ← true if any check failed critically
  block_reason,
  audit_id,               ← correlates with structured audit log line
  latency_ms              ← per-phase breakdown
}
```

### Blocking Policy
| Condition | Default Behaviour |
|---|---|
| Injection risk ≥ `SECURITY_INJECTION_THRESHOLD` (0.7) | **Block** — HTTP 400 |
| PII found + `SECURITY_BLOCK_ON_PII=true` | **Block** — HTTP 400 |
| PII found + `SECURITY_BLOCK_ON_PII=false` | **Mask and continue** |
| File fails any validation | **Block** — HTTP 400 |
| Malicious file signature | **Block** — HTTP 400 |

### Prometheus Metrics (security namespace)
```
rag_security_checks_total               # Counter: total security checks
rag_security_blocks_total{reason}       # Counter: blocks by reason
rag_security_pii_entities{buckets}      # Histogram: entities per request
rag_security_pii_by_type_total{type}    # Counter: detections by entity type
rag_security_injection_risk{buckets}    # Histogram: risk score distribution
rag_security_injection_by_category      # Counter: attacks by category
rag_security_file_rejections{reason}    # Counter: file rejections by reason
rag_security_file_size_bytes{buckets}   # Histogram: accepted file sizes
rag_security_latency_seconds{buckets}   # Histogram: security pipeline latency
rag_security_presidio_available         # Gauge: 1=Presidio, 0=regex fallback
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v2/query` | Main RAG query (JSON body: `RAGRequest`) |
| `POST` | `/api/v2/upload` | Secure file upload (multipart/form-data, field: `files`) |
| `POST` | `/api/v2/admin/index-bm25` | Rebuild BM25 index after new document ingestion |
| `GET` | `/api/v2/health` | Health check with component status |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |
| `GET` | `/docs` | Swagger UI |

---

## Query Type → Strategy Mapping

| Query Type | Strategy | Graph | Verify | Level | Max Hops |
|---|---|---|---|---|---|
| SIMPLE | HYBRID | No | No | — | 1 |
| MULTI_HOP | HYBRID | Yes | Yes | standard | 2 |
| KNOWLEDGE_GRAPH | HYBRID_GRAPH | Yes | Yes | standard | 3 |
| COMPLIANCE | HYBRID | No | Yes | **strict** | 1 |
| RESEARCH | HYBRID_GRAPH | Yes | Yes | standard | 2 |

---

## Integration Points with Existing RAG

- **ChromaDB**: `HybridRetriever` queries existing collection — no schema changes required
- **PDF/OCR/Chunking**: Call `POST /api/v2/admin/index-bm25` after ingestion to sync BM25 index
- **Security → Pipeline**: `effective_query` (PII-masked) is passed to all downstream phases; original query is never forwarded after masking
- **Observability**: Security phase emits `security_audit` structured log and updates 9 Prometheus metrics; LangSmith receives a `security` child span on every request

---

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env — add OPENAI_API_KEY, NEO4J_PASSWORD at minimum

# 2. Start all services
docker compose up -d

# 3. Verify
curl http://localhost:8080/api/v2/health

# 4. Query
curl -X POST http://localhost:8080/api/v2/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the GDPR data retention requirements?"}'

# 5. Upload a file
curl -X POST http://localhost:8080/api/v2/upload \
  -F "files=@document.pdf"

# 6. View metrics
open http://localhost:9091   # Prometheus
open http://localhost:3000   # Grafana (admin/admin)
open http://localhost:7474   # Neo4j Browser
```
