.PHONY: setup venv install run clean test lint format help use-openai use-ollama

# Detect operating system
ifeq ($(OS),Windows_NT)
	# Windows settings
	PYTHON := python
	VENV := venv
	VENV_BIN := $(VENV)\Scripts
	VENV_ACTIVATE := $(VENV_BIN)\activate.bat
	SEP := \\
	RM := rmdir /s /q
	MKDIR := mkdir
	# Command prefix for Windows
	CMD_PREFIX := 
else
	# Unix-like settings (macOS and Linux)
	PYTHON := python3
	VENV := venv
	VENV_BIN := $(VENV)/bin
	SEP := /
	RM := rm -rf
	MKDIR := mkdir -p
	
	# Check if macOS
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		# macOS specific settings
		CMD_PREFIX := . $(VENV_BIN)/activate &&
	else
		# Linux specific settings
		CMD_PREFIX := . $(VENV_BIN)/activate &&
	endif
endif

# Default target
.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make setup      - Create virtual environment and install requirements"
	@echo "  make venv       - Create virtual environment only"
	@echo "  make install    - Install requirements only"
	@echo "  make run        - Run the RAG API server"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Remove virtual environment and cached files"
	@echo "  make lint       - Run linter"
	@echo "  make format     - Format code using black"
	@echo "  make use-openai - Switch to OpenAI as provider"
	@echo "  make use-ollama - Switch to Ollama as provider"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Detected environment: $(if $(OS),Windows,$(if $(filter Darwin,$(UNAME_S)),macOS,Linux))"

setup: venv install

venv:
	@echo "Creating virtual environment..."
ifeq ($(OS),Windows_NT)
	@if not exist $(VENV) $(PYTHON) -m venv $(VENV)
else
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
endif
	@echo "Virtual environment created at $(VENV)/"

install: venv
	@echo "Installing requirements..."
ifeq ($(OS),Windows_NT)
	@$(VENV_BIN)\pip install --upgrade pip
	@$(VENV_BIN)\pip install -r requirements.txt
else
	@$(CMD_PREFIX) pip install --upgrade pip
	@$(CMD_PREFIX) pip install -r requirements.txt
endif
	@echo "Requirements installed."

run: venv
	@echo "Starting RAG API server..."
ifeq ($(OS),Windows_NT)
	@$(VENV_BIN)\python main.py
else
	@$(CMD_PREFIX) python main.py
endif

test: venv
	@echo "Running tests..."
ifeq ($(OS),Windows_NT)
	@$(VENV_BIN)\pytest tests/
else
	@$(CMD_PREFIX) pytest tests/
endif

clean:
	@echo "Cleaning up..."
ifeq ($(OS),Windows_NT)
	@if exist $(VENV) $(RM) $(VENV)
	@if exist .pytest_cache $(RM) .pytest_cache
	@if exist __pycache__ $(RM) __pycache__
	@if exist .coverage del .coverage
	@for /r %%i in (__pycache__) do @if exist "%%i" $(RM) "%%i"
	@for /r %%i in (*.pyc) do @if exist "%%i" del "%%i"
else
	@$(RM) $(VENV) .pytest_cache __pycache__ .coverage
	@find . -type d -name "__pycache__" -exec $(RM) {} +
	@find . -type d -name "*.egg-info" -exec $(RM) {} +
	@find . -type f -name "*.pyc" -delete
endif
	@echo "Cleanup complete."

lint: venv
	@echo "Running linter..."
ifeq ($(OS),Windows_NT)
	@$(VENV_BIN)\pip install pylint
	@$(VENV_BIN)\pylint src/
else
	@$(CMD_PREFIX) pip install pylint
	@$(CMD_PREFIX) pylint src/
endif

format: venv
	@echo "Formatting code..."
ifeq ($(OS),Windows_NT)
	@$(VENV_BIN)\pip install black
	@$(VENV_BIN)\black src/ tests/
else
	@$(CMD_PREFIX) pip install black
	@$(CMD_PREFIX) black src/ tests/
endif

use-openai: venv
	@echo "Switching to OpenAI provider..."
ifeq ($(OS),Windows_NT)
	@$(VENV_BIN)\python switch_provider.py openai
else
	@$(CMD_PREFIX) python switch_provider.py openai
endif
	@echo "Now using OpenAI as the LLM provider."
	@echo "Don't forget to set your OPENAI_API_KEY in the .env file!"

use-ollama: venv
	@echo "Switching to Ollama provider..."
ifeq ($(OS),Windows_NT)
	@$(VENV_BIN)\python switch_provider.py ollama
else
	@$(CMD_PREFIX) python switch_provider.py ollama
endif
	@echo "Now using Ollama as the LLM provider."
	@echo "Make sure Ollama is installed and running: https://ollama.ai/download" 