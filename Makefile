.PHONY: help install test test-unit test-integration test-coverage lint format clean docs

help:
	@echo "kv-planner - Development Commands"
	@echo ""
	@echo "  install          Install package and dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  lint             Run linters (flake8, mypy)"
	@echo "  format           Format code (black, isort)"
	@echo "  clean            Clean build artifacts"
	@echo "  docs             Build documentation"

install:
	pip install -e ".[dev,hf]"

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-coverage:
	pytest tests/ --cov=kv_planner --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov dist build *.egg-info

docs:
	cd docs && mkdocs build

serve-docs:
	cd docs && mkdocs serve
