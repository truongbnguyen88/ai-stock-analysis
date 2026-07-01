.PHONY: install check test lint typecheck check-math format clean ui pull-models verify-models refresh-filings rag-eval

install:  ## editable install with dev tooling
	pip install -e ".[dev]"

check: lint typecheck check-math test  ## run the full gate (must be green to advance a phase)

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy

check-math:  ## flag GitHub-MathJax render hazards in docs/ (see the github-math skill)
	python scripts/check_github_math.py docs/*.md

format:  ## auto-format and apply safe lint fixes
	ruff format src tests
	ruff check --fix src tests

ui:  ## launch the Streamlit chat frontend (browser opens automatically)
	PYTHONPATH=src streamlit run ui/chat_app.py

pull-models:  ## download the latest CI-trained models + conformal.json into outputs/models/
	@mkdir -p outputs/models
	@REPO=$$(git config --get remote.origin.url | sed -E 's#.*github\.com[:/]##; s#\.git$$##'); \
	URL="https://github.com/$$REPO/releases/download/models-latest/models.tar.gz"; \
	echo "Downloading $$URL"; \
	curl -fL "$$URL" -o outputs/models/models.tar.gz || { \
	  echo "✗ no 'models-latest' release yet (run the retrain workflow first), or download failed"; \
	  exit 1; }
	tar -xzf outputs/models/models.tar.gz -C outputs/models
	@rm -f outputs/models/models.tar.gz
	@echo "✓ Pulled latest models + conformal.json into outputs/models/"

verify-models:  ## structural sanity-check of the trained artifacts (CI promote gate)
	PYTHONPATH=src python -m stock_agent verify-models

refresh-filings:  ## RAG quarterly refresh: pull new SEC filings + incrementally embed them (local)
	PYTHONPATH=src python -m stock_agent documents refresh --all --months 6

rag-eval:  ## advanced-RAG: local retrieval lattice (dense/reranked/hybrid/hybrid+rerank) — NOT in `make check`
	PYTHONPATH=src python -m stock_agent rag eval \
	  --queries configs/rag_eval_queries.json \
	  --systems dense,reranked,hybrid,hybrid+rerank --diagnostic \
	  --report outputs/rag_eval/lattice.json

rag-eval-multistep:  ## advanced-RAG A4: multi-hop union coverage vs single-shot — PAID (real LLM) — NOT in `make check`
	PYTHONPATH=src python -m stock_agent rag eval-multistep \
	  --queries configs/rag_eval_multistep.json \
	  --report outputs/rag_eval/multistep.json

rag-gen-multistep:  ## advanced-RAG A6.0: mine the graph into the stratified multi-hop benchmark — local, $0 — NOT in `make check`
	PYTHONPATH=src python -m stock_agent rag gen-multistep \
	  --universe configs/graph_universe.txt \
	  --out configs/rag_eval_multistep_generated.json \
	  --supply-report outputs/rag_eval/multistep_supply.json \
	  --split-test-frac 0.3

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
