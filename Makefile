.PHONY: install check test lint typecheck format clean ui pull-models verify-models

install:  ## editable install with dev tooling
	pip install -e ".[dev]"

check: lint typecheck test  ## run the full gate (must be green to advance a phase)

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy

format:  ## auto-format and apply safe lint fixes
	ruff format src tests
	ruff check --fix src tests

ui:  ## launch the Streamlit chat frontend (browser opens automatically)
	PYTHONPATH=src streamlit run ui/chat_app.py

pull-models:  ## download the latest CI-trained models from GitHub into outputs/models/
	@mkdir -p outputs/models
	@REPO=$$(git config --get remote.origin.url | sed -E 's#.*github\.com[:/]##; s#\.git$$##'); \
	URL="https://github.com/$$REPO/releases/download/models-latest/models.tar.gz"; \
	echo "Downloading $$URL"; \
	curl -fL "$$URL" -o outputs/models/models.tar.gz || { \
	  echo "✗ no 'models-latest' release yet (run the retrain workflow first), or download failed"; \
	  exit 1; }
	tar -xzf outputs/models/models.tar.gz -C outputs/models
	@rm -f outputs/models/models.tar.gz
	@echo "✓ Pulled latest models into outputs/models/"

verify-models:  ## structural sanity-check of the trained artifacts (CI promote gate)
	PYTHONPATH=src python -m stock_agent verify-models

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
