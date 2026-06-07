# LuminaAI Desktop - Development Makefile
.PHONY: help install dev build test clean docker

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)LuminaAI Desktop Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install all dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(BLUE)Installing Node.js dependencies...$(NC)"
	npm install
	@echo "$(GREEN) Installation complete!$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install -r requirements.txt pytest flake8 black
	npm install
	@echo "$(GREEN) Development setup complete!$(NC)"

dev: ## Start development environment
	@echo "$(BLUE)Starting LuminaAI development environment...$(NC)"
	npm run dev

dev-backend: ## Start only Python backend
	@echo "$(BLUE)Starting Python backend (lumina_desktop.py)...$(NC)"
	python lumina_desktop.py

dev-frontend: ## Start only Electron frontend
	@echo "$(BLUE)Starting Electron frontend...$(NC)"
	npm run dev:electron

build: ## Build the application
	@echo "$(BLUE)Building LuminaAI Desktop...$(NC)"
	npm run build
	@echo "$(GREEN) Build complete!$(NC)"

build-all: ## Build for all platforms
	@echo "$(BLUE)Building for all platforms...$(NC)"
	npm run dist:all
	@echo "$(GREEN) Multi-platform build complete!$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running Python tests...$(NC)"
	python -c "import train; print(' train.py')"
	python -c "import fine_tune; print(' fine_tune.py')"
	python -c "import ChatAI; print(' ChatAI.py')"
	python -c "import buildapp; print(' buildapp.py')"
	python -c "import lumina_desktop; print(' lumina_desktop.py')"
	@echo "$(BLUE)Running Node.js tests...$(NC)"
	npm test
	@echo "$(GREEN) All tests passed!$(NC)"

test-python: ## Test Python scripts only
	@echo "$(BLUE)Testing Python scripts...$(NC)"
	python -c "import train; print(' train.py imports successfully')"
	python -c "import fine_tune; print(' fine_tune.py imports successfully')"
	python -c "import ChatAI; print(' ChatAI.py imports successfully')"
	python -c "import buildapp; print(' buildapp.py imports successfully')"
	python -c "import lumina_desktop; print(' lumina_desktop.py imports successfully')"

lint: ## Run code linting
	@echo "$(BLUE)Linting Python code...$(NC)"
	flake8 *.py --max-line-length=127 || echo "$(YELLOW)  Python linting issues found$(NC)"
	@echo "$(BLUE)Linting JavaScript code...$(NC)"
	npm run lint || echo "$(YELLOW)  JavaScript linting issues found$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting Python code...$(NC)"
	black *.py
	@echo "$(BLUE)Formatting JavaScript code...$(NC)"
	npm run format
	@echo "$(GREEN) Code formatting complete!$(NC)"

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf dist/ build/ node_modules/.cache/
	rm -rf __pycache__/ *.pyc
	npm run clean
	@echo "$(GREEN) Clean complete!$(NC)"

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build
	@echo "$(GREEN) Docker build complete!$(NC)"

docker-up: ## Start Docker containers
	@echo "$(BLUE)Starting Docker containers...$(NC)"
	docker-compose up -d
	@echo "$(GREEN) Docker containers started!$(NC)"

docker-dev: ## Start development Docker environment
	@echo "$(BLUE)Starting development Docker environment...$(NC)"
	docker-compose -f docker-compose.dev.yml up
	@echo "$(GREEN) Development environment started!$(NC)"

docker-down: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	docker-compose down
	@echo "$(GREEN) Docker containers stopped!$(NC)"

docker-logs: ## View Docker logs
	@echo "$(BLUE)Docker container logs:$(NC)"
	docker-compose logs -f

setup: ## First-time setup
	@echo "$(BLUE) Setting up LuminaAI Desktop for the first time...$(NC)"
	@echo "$(BLUE)Creating directories...$(NC)"
	mkdir -p models data logs assets
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(MAKE) install-dev
	@echo "$(BLUE)Running initial tests...$(NC)"
	$(MAKE) test-python
	@echo "$(GREEN) Setup complete! Run 'make dev' to start developing.$(NC)"

info: ## Show project information
	@echo "$(BLUE)LuminaAI Desktop Project Information$(NC)"
	@echo "$(GREEN)Python Scripts:$(NC)"
	@echo "   train.py - Model training"
	@echo "   fine_tune.py - Model fine-tuning"  
	@echo "   ChatAI.py - Chat interface"
	@echo "   buildapp.py - Application builder"
	@echo "   lumina_desktop.py - Main desktop server"
	@echo ""
	@echo "$(GREEN)Available Commands:$(NC)"
	@echo "   make dev - Start development"
	@echo "   make build - Build application"
	@echo "   make test - Run tests"
	@echo "   make docker-dev - Start with Docker"

# Default target
.DEFAULT_GOAL := help