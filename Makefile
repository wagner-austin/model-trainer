SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

# Paths (relative to this Model-Trainer root)
PY_DIR := server
UI_DIR := ui

# Docker Compose project name (isolates this stack)
PROJECT_NAME := modeltrainer
COMPOSE := docker compose -p $(PROJECT_NAME) -f .\docker-compose.yml

.PHONY: help lint guards test check start stop restart clean

help:
	@echo "Targets:"
	@echo "  make lint        - Ruff --fix, format, mypy strict, guard checks"
	@echo "  make test        - Run tests"
	@echo "  make check       - Lint, then run tests"
	@echo "  make start       - Build and start Docker stack (Redis, API, Worker)"
	@echo "  make stop        - Gracefully stop Docker stack"
	@echo "  make restart     - Restart Docker stack"
	@echo "  make clean       - Remove this stack's containers/images/volumes, then start"

lint: guards
	python .\tools\lint_runner.py

guards:
	python .\scripts\guard.py

test:
	if (Test-Path "$(PY_DIR)\pyproject.toml") { Write-Host "[test-python] pytest with coverage (branches)" -ForegroundColor Cyan; Push-Location "$(PY_DIR)"; poetry run pytest --cov=model_trainer --cov-branch --cov-report=term-missing -v; Pop-Location; } else { Write-Host "[test-python] Skipped: $(PY_DIR) not initialized" -ForegroundColor Yellow; }

# Run lint then tests without recursive make chatter
check: lint test

start:
	if (-not (Test-Path ".env") -and (Test-Path ".env.example")) { Write-Host "[env] Creating .env from .env.example" -ForegroundColor Yellow; Copy-Item ".env.example" ".env" }
	if (-not (Test-Path ".\artifacts")) { New-Item -ItemType Directory ".\artifacts" | Out-Null }
	if (-not (Test-Path ".\runs")) { New-Item -ItemType Directory ".\runs" | Out-Null }
	if (-not (Test-Path ".\logs")) { New-Item -ItemType Directory ".\logs" | Out-Null }
	Write-Host "[compose] Starting stack: $(PROJECT_NAME)" -ForegroundColor Cyan; $(COMPOSE) up -d --build

stop:
	Write-Host "[compose] Stopping stack: $(PROJECT_NAME)" -ForegroundColor Yellow; $(COMPOSE) stop

restart:
	$(MAKE) stop; $(MAKE) start

clean:
	Write-Host "[compose] Removing containers/images/volumes for project $(PROJECT_NAME)" -ForegroundColor Red; $(COMPOSE) down --volumes --rmi all --remove-orphans || $$True; $(MAKE) start
