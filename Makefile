.PHONY: help install verify notebook lab clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make verify     - Verify Python setup"
	@echo "  make notebook   - Start Jupyter Notebook"
	@echo "  make lab        - Start Jupyter Lab (recommended)"
	@echo "  make clean      - Remove cache files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

verify:
	python verify_setup.py

notebook:
	jupyter notebook

lab:
	jupyter lab

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete

