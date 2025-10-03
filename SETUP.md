# Setup Guide - AI & ML Learning Plan

Complete setup guide for your 52-week learning journey. This repository is for **learning**, not production.

## What This Repo Is For

✅ **Learning** - Jupyter notebooks, experiments, understanding  
✅ **Exploring** - Try ideas, make mistakes, iterate  
✅ **Documenting** - Markdown notes, visualizations  
✅ **Building intuition** - Code from scratch  

❌ **NOT for** - Production code, unit tests, CI/CD, strict code quality

---

## Quick Start (2 Minutes)

```bash
# 1. Setup environment
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify setup
python verify_setup.py

# 4. Start learning!
jupyter lab
```

**First notebook**: Open `journal/weeks/WEEK_01/W01_day_by_day.ipynb`

---

## Detailed Setup

### Prerequisites

- **Python 3.11+** (you have 3.13+ which is great!)
- **pip** (Python package installer)
- **git** (for version control)
- **Terminal** (macOS/Linux) or **PowerShell** (Windows)

### Step 1: Install Python 3.11+

#### macOS
```bash
# Check current version
python3 --version

# If needed, install with Homebrew
brew install python@3.11

# Or use pyenv for version management
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
- Download from [python.org](https://www.python.org/downloads/)
- Or use Chocolatey: `choco install python --version=3.11.0`

### Step 2: Create Virtual Environment

Virtual environments keep dependencies isolated and prevent conflicts.

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Your prompt should now show (.venv)
```

**Important**: Always activate your virtual environment before working!

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies (~40 packages)
pip install -r requirements.txt
```

This installs:
- **Core**: NumPy, Pandas, Matplotlib, SciPy, scikit-learn
- **Jupyter**: jupyter, ipykernel, ipywidgets
- **Deep Learning**: PyTorch, Lightning
- **Experimentation**: Weights & Biases, FastAPI
- **Recommender Systems**: implicit (lightfm incompatible with Python 3.13+)
- **LLM & Agents**: LangChain, OpenAI, instructor, sentence-transformers
- **Vector Search**: FAISS
- **Utilities**: requests, tqdm, python-dotenv

**Note**: Installation takes 3-5 minutes (PyTorch is ~250MB).

### Step 4: Verify Setup

```bash
python verify_setup.py
```

Expected output:
```
✅ Python 3.13.7
✅ numpy     (v2.3.3)
✅ pandas    (v2.3.3)
...
✅ Setup verified! You're ready to start learning.
```

### Step 5: Start Jupyter

```bash
# Jupyter Lab (recommended - modern interface)
jupyter lab

# OR Jupyter Notebook (classic interface)
jupyter notebook
```

Your browser opens automatically at `http://localhost:8888`

Navigate to: `journal/weeks/WEEK_01/W01_day_by_day.ipynb`

---

## Using the Makefile

Convenient shortcuts for common commands:

```bash
make help       # Show all commands
make install    # Install dependencies
make verify     # Verify setup
make lab        # Start Jupyter Lab
make notebook   # Start Jupyter Notebook
make clean      # Remove cache files
```

---

## Troubleshooting

### "python3.11 not found"
- Check version: `python3 --version`
- Try: `python3` or `python` instead of `python3.11`
- Add Python to your PATH
- Your Python 3.13 works perfectly (newer than 3.11)

### "pip: command not found"
```bash
python3 -m ensurepip --upgrade
```

### PyTorch Installation Issues

**CPU-only** (faster install, smaller size):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA)**: See [pytorch.org/get-started](https://pytorch.org/get-started/locally/)

### Import Errors in Jupyter

Ensure Jupyter uses the correct kernel:
```bash
# Install kernel in your venv
python -m ipykernel install --user --name=ai-ml-env

# In Jupyter: Kernel > Change Kernel > ai-ml-env
```

### Slow Installation

Use a faster PyPI mirror:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### lightfm Not Installing

**Expected**: lightfm is incompatible with Python 3.13 (you won't need it until Week 23).
- It's marked optional in `requirements.txt`
- By Week 23, it may be updated, or you can use alternatives

---

## Environment Variables

For API keys and secrets, create a `.env` file:

```bash
# .env (create this file - already in .gitignore)
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

---

## Learning Workflow

### Daily Routine (Monday - Friday, ~60 min)

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Start Jupyter Lab
jupyter lab

# 3. Work on current week
# - Open notebooks in journal/weeks/WEEK_XX/
# - Code, experiment, visualize
# - Take notes in markdown cells

# 4. Save and commit (end of session)
git add .
git commit -m "Week X: [what you learned]"
git push
```

### Weekly Rhythm

- **Monday**: Theory (videos/reading) ~60 min
- **Tuesday**: Papers (skim & notes) ~60 min
- **Wednesday**: Code from scratch ~60 min
- **Thursday**: Extend/experiment ~60 min
- **Friday**: Mini-deliverable (writeup/result) ~60 min

### What You'll Create Each Week

```
journal/weeks/WEEK_XX/
├── README.md              # Week notes and insights
├── *.ipynb                # Jupyter notebooks (main work)
├── *.py                   # Python utilities (if needed)
└── reading_list.md        # Papers and resources
```

---

## What's Installed (Package Overview)

### Phase 1: Math & Foundations (Weeks 1-12)
- NumPy, SciPy (linear algebra, optimization)
- Pandas (data manipulation)
- Matplotlib, Seaborn (visualization)
- scikit-learn (traditional ML)

### Phase 2: Deep Learning (Weeks 13-20)
- PyTorch (neural networks)
- Lightning (training framework)
- torchvision (computer vision)

### Phase 3: Recommender Systems (Weeks 21-32)
- implicit (collaborative filtering)
- FAISS (vector search)
- lightfm (optional - Python 3.13 incompatible)

### Phase 4: Agentic AI (Weeks 33-42)
- LangChain (agent frameworks)
- OpenAI API (LLM access)
- instructor (structured outputs)
- sentence-transformers (embeddings)

### Tools Throughout
- Jupyter Lab (interactive development)
- Weights & Biases (experiment tracking)
- FastAPI (model serving - later phases)
- SQLAlchemy (data persistence - later phases)

**Total**: ~40 packages covering the entire 52-week curriculum

---

## Learning Philosophy

### ✅ Focus On

- **Understanding deeply** - Why does this work?
- **Coding from scratch** - Implement before using libraries
- **Visualizing concepts** - Plots build intuition
- **Taking notes** - Document as you learn
- **Experimenting freely** - Try variations
- **Reviewing regularly** - Spaced repetition

### ❌ Don't Worry About

- Writing tests (focus on understanding first)
- Code formatting (clarity over style)
- Production patterns (learn first, optimize later)
- Perfect code (messy notebooks are okay!)
- Type hints (not needed for learning)
- Comprehensive documentation (simple comments are enough)

> **Key Principle**: Learning beats optimization. Understand deeply, then move forward.

---

## Project Structure

```
ai-ml-learning-plan/
├── README.md                    # 52-week learning plan
├── SETUP.md                     # This file
├── QUICK_REFERENCE.md           # Daily command cheat sheet
├── requirements.txt             # All dependencies
├── Makefile                     # Convenient commands
├── verify_setup.py              # Setup verification
│
├── journal/weeks/               # Your work goes here!
│   └── WEEK_XX/
│       ├── README.md            # Week notes
│       ├── *.ipynb              # Jupyter notebooks
│       ├── *.py                 # Python utilities
│       └── reading_list.md      # Papers/resources
│
├── courses/                     # Course references
├── certs/                       # Certification guides
└── scripts/
    └── update_badge.py          # Progress badge updater
```

---

## Next Steps

1. ✅ **Verify setup**: Run `python verify_setup.py`
2. 📚 **Read the plan**: Open `README.md` for the full 52-week curriculum
3. 🚀 **Start Jupyter**: Run `jupyter lab`
4. 📝 **Open Week 1**: Navigate to `journal/weeks/WEEK_01/W01_day_by_day.ipynb`
5. 🎯 **Follow the rhythm**: Theory → Papers → Code → Experiment → Deliverable

---

## Helpful Resources

**Documentation**
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Jupyter Lab](https://jupyterlab.readthedocs.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html)

**In This Repo**
- `README.md` - Full 52-week plan with phases
- `QUICK_REFERENCE.md` - Command cheat sheet for daily use
- `courses/` - Course references and links
- `certs/` - Certification study guides

---

**Happy Learning!** 🚀

*Consistency > Intensity. Ship something small every Friday.*
