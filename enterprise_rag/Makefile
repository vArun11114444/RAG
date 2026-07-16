# ── Enterprise Agentic RAG — Developer Makefile ───────────────────────────────
.PHONY: help install dev-install test test-security test-unit test-cov \
        lint format type-check run run-dev docker-up docker-down \
        docker-logs clean spacy-model

PYTHON  := python3
PIP     := pip
PYTEST  := pytest
APP     := app.main:app

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Installation ──────────────────────────────────────────────────────────────
install:        ## Install production dependencies
	$(PIP) install -r requirements.txt

dev-install:    ## Install production + dev dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install ruff mypy black

spacy-model:    ## Download spaCy model (required for Presidio + KG extraction)
	$(PYTHON) -m spacy download en_core_web_sm

# ── Testing ───────────────────────────────────────────────────────────────────
test:           ## Run all tests
	$(PYTEST) tests/ -v

test-security:  ## Run security layer tests only
	$(PYTEST) tests/test_security.py -v -m security

test-unit:      ## Run unit tests only (no external services)
	$(PYTEST) tests/ -v -m "not integration"

test-cov:       ## Run all tests with HTML coverage report
	$(PYTEST) tests/ --cov=app --cov-report=html --cov-report=term-missing -v
	@echo "Coverage report: htmlcov/index.html"

test-fast:      ## Run tests excluding slow ones
	$(PYTEST) tests/ -v -m "not slow and not integration"

# ── Code quality ──────────────────────────────────────────────────────────────
lint:           ## Lint with ruff
	ruff check app/ tests/

format:         ## Format with black + ruff
	black app/ tests/
	ruff check --fix app/ tests/

type-check:     ## Type check with mypy
	mypy app/ --ignore-missing-imports

# ── Running ───────────────────────────────────────────────────────────────────
run:            ## Run production server (2 workers)
	uvicorn $(APP) --host 0.0.0.0 --port 8080 --workers 2

run-dev:        ## Run dev server with auto-reload
	uvicorn $(APP) --host 0.0.0.0 --port 8080 --reload --log-level debug

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:      ## Start all services with docker compose
	docker compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	@echo "API:        http://localhost:8080"
	@echo "Prometheus: http://localhost:9091"
	@echo "Grafana:    http://localhost:3000 (admin/admin)"
	@echo "Neo4j:      http://localhost:7474"

docker-down:    ## Stop all services
	docker compose down

docker-logs:    ## Tail API logs
	docker compose logs -f api

docker-rebuild: ## Rebuild and restart API container only
	docker compose up -d --build api

docker-clean:   ## Remove all containers and volumes
	docker compose down -v

# ── Utilities ─────────────────────────────────────────────────────────────────
health:         ## Check API health
	curl -s http://localhost:8080/api/v2/health | python3 -m json.tool

index-bm25:     ## Rebuild BM25 index (requires API running)
	curl -s -X POST http://localhost:8080/api/v2/admin/index-bm25 \
		-H "Content-Type: application/json" -d '[]'

clean:          ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage .mypy_cache

env-check:      ## Validate .env file has all required variables
	@$(PYTHON) -c "
import os, sys
from dotenv import load_dotenv
load_dotenv()
required = [
    'OPENAI_API_KEY', 'NEO4J_PASSWORD',
    'CHROMA_HOST', 'CHROMA_PORT',
]
missing = [k for k in required if not os.getenv(k)]
if missing:
    print('Missing required env vars:', missing)
    sys.exit(1)
print('All required env vars present')
"
