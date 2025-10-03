# Python & Jupyter Setup Guide

This guide will help you set up your Python environment for the AI & ML Learning Plan.

## Prerequisites

- **Python 3.11+** (recommended: 3.11.0)
- **pip** (Python package installer)
- **git** (for version control)
- **macOS, Linux, or Windows** with terminal access

## Quick Start

```bash
# 1. Clone the repository (if from GitHub)
git clone <your-repo-url>
cd ai-ml-learning-plan

# 2. Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify setup
python verify_setup.py

# 5. Start learning!
jupyter lab
```

## Detailed Setup

### 1. Install Python 3.11+

#### macOS
```bash
# Using Homebrew
brew install python@3.11

# Using pyenv (recommended for managing multiple versions)
brew install pyenv
pyenv install 3.11.0
pyenv local 3.11.0
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

#### Windows
Download from [python.org](https://www.python.org/downloads/) or use:
```powershell
# Using Chocolatey
choco install python --version=3.11.0
```

### 2. Virtual Environment Setup

Virtual environments keep dependencies isolated and prevent conflicts.

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Your prompt should now show (.venv)
```

**Important**: Always activate your virtual environment before working on the project!

### 3. Install Dependencies

The project uses `requirements.txt` for dependency management:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

This will install:
- **Core**: numpy, pandas, matplotlib, scipy, scikit-learn
- **Jupyter**: jupyter, ipykernel, ipywidgets
- **Deep Learning**: PyTorch, Lightning
- **MLOps**: Weights & Biases, FastAPI
- **Recommender Systems**: implicit, lightfm
- **LLM & Agents**: langchain, openai
- **Utilities**: requests, tqdm, python-dotenv

### 4. Verify Setup

Run the verification script to ensure everything is installed correctly:

```bash
python verify_setup.py
```

You should see:
```
✅ Python 3.11.x
✅ numpy                (v1.24.0)
✅ pandas               (v2.0.0)
✅ matplotlib           (v3.7.0)
...
✅ Setup verified! You're ready to start learning.
```

### 5. Running Jupyter Notebooks

Start Jupyter Lab (recommended) or Jupyter Notebook:

```bash
# Jupyter Lab (modern interface)
jupyter lab

# OR Jupyter Notebook (classic interface)
jupyter notebook
```

Your browser will open automatically. Navigate to `journal/weeks/WEEK_01/` to start.

## Using the Makefile

The project includes a Makefile for convenience:

```bash
make help       # Show all available commands
make install    # Install dependencies
make verify     # Verify setup
make lab        # Start Jupyter Lab
make notebook   # Start Jupyter Notebook
make clean      # Remove cache files
```

## Troubleshooting

### "python3.11 not found"
- Ensure Python 3.11+ is installed: `python3 --version`
- Try `python3` or `python` instead of `python3.11`
- Add Python to your PATH

### "pip: command not found"
```bash
# Install pip
python3 -m ensurepip --upgrade
```

### PyTorch Installation Issues
For CPU-only (smaller, faster to install):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For GPU (CUDA) support, see [pytorch.org](https://pytorch.org/get-started/locally/)

### Import Errors in Jupyter
Ensure Jupyter is using the correct kernel:
```bash
# Install kernel in your venv
python -m ipykernel install --user --name=ai-ml-env

# In Jupyter, select: Kernel > Change Kernel > ai-ml-env
```

### Slow Installation
Use a faster mirror:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Environment Variables

For API keys and secrets, create a `.env` file (already in .gitignore):

```bash
# .env (create this file)
OPENAI_API_KEY=your-key-here
WANDB_API_KEY=your-key-here
HF_TOKEN=your-token-here
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

## Learning Workflow

1. **Activate environment**: `source .venv/bin/activate`
2. **Start Jupyter**: `jupyter lab`
3. **Work on notebooks**: Complete weekly exercises
4. **Take notes**: Document learnings in markdown files
5. **Experiment**: Try different approaches and parameters
6. **Review**: Look back at previous weeks' work

## Project Organization

Your weekly work goes in:
```
journal/weeks/WEEK_XX/
  ├── README.md              # Week overview and notes
  ├── *.ipynb                # Jupyter notebooks for experiments
  ├── *.py                   # Python utilities/modules
  └── reading_list.md        # Papers and resources
```

## Next Steps

Once setup is complete:
1. Open `journal/weeks/WEEK_01/W01_PCA_via_SVD.ipynb`
2. Read through the README.md for the full 52-week plan
3. Check the `courses/` and `certs/` directories for resources
4. Follow the weekly rhythm: theory → papers → code → experiment → deliverable

## Helpful Resources

- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Jupyter Lab Documentation](https://jupyterlab.readthedocs.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/)

---

**Need help?** Check the documentation for specific packages or search for tutorials online.
