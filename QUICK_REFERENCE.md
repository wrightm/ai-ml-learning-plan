# Quick Reference - AI/ML Learning

A one-page cheat sheet for your daily learning workflow.

## 🚀 Getting Started (First Time)

```bash
# 1. Create virtual environment
python3.11 -m venv .venv

# 2. Activate
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate          # Windows

# 3. Install
pip install -r requirements.txt

# 4. Verify
python verify_setup.py

# 5. Start learning
jupyter lab
```

## 📋 Daily Commands

```bash
# Environment
source .venv/bin/activate       # Activate venv (ALWAYS DO THIS FIRST!)
deactivate                      # Deactivate venv
echo $VIRTUAL_ENV              # Check if venv is active

# Jupyter
jupyter lab                     # Start Jupyter Lab (recommended)
jupyter notebook                # Start Jupyter Notebook (classic)

# Using Makefile (easier)
make help                       # Show all commands
make install                    # Install dependencies
make verify                     # Check setup
make lab                        # Start Jupyter Lab
make notebook                   # Start Jupyter Notebook
make clean                      # Remove cache files
```

## 📓 Jupyter Tips

### Launching and Setup
```bash
# Start Jupyter Lab
jupyter lab

# Install custom kernel (if needed)
python -m ipykernel install --user --name=ai-ml-env

# List kernels
jupyter kernelspec list

# Convert notebook to Python script
jupyter nbconvert --to python notebook.ipynb
```

### Jupyter Magic Commands
```python
# Show plots inline
%matplotlib inline

# Auto-reload modules (useful for .py files)
%load_ext autoreload
%autoreload 2

# Time execution
%time expression                # Time single expression
%%time                          # Time entire cell

# Show variable info
%whos                           # List all variables
%whos numpy.ndarray             # List variables of specific type

# Run shell commands
!pip install package-name
!ls -la

# Load Python file into cell
%load filename.py

# Show matplotlib plots inline
%config InlineBackend.figure_format = 'retina'  # High-res plots
```

### Keyboard Shortcuts
**Command Mode** (press Esc):
- `A` - Insert cell above
- `B` - Insert cell below
- `D D` - Delete cell
- `M` - Change to markdown
- `Y` - Change to code
- `Shift+Enter` - Run cell, select below
- `Ctrl+Enter` - Run cell, stay selected
- `Alt+Enter` - Run cell, insert below

**Edit Mode** (press Enter):
- `Tab` - Autocomplete
- `Shift+Tab` - Show docstring
- `Ctrl+]` - Indent
- `Ctrl+[` - Dedent
- `Cmd+/` or `Ctrl+/` - Toggle comment

## 📦 Package Management

```bash
# Install new package
pip install package-name

# Install specific version
pip install package-name==1.2.3

# Update package
pip install --upgrade package-name

# Update all
pip install --upgrade -r requirements.txt

# Show installed packages
pip list

# Show package info
pip show numpy

# Uninstall
pip uninstall package-name

# Check for outdated packages
pip list --outdated
```

## 🔧 Common Python Imports

```python
# Essential imports for ML/AI work
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Traditional ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Utilities
from pathlib import Path
import os
from tqdm import tqdm
```

## 📊 Quick Data Exploration

```python
# Pandas basics
df = pd.read_csv('data.csv')
df.head()                       # First 5 rows
df.info()                       # Column types and non-null counts
df.describe()                   # Statistical summary
df.shape                        # (rows, columns)
df.columns                      # Column names
df.isnull().sum()              # Count missing values

# NumPy basics
arr = np.array([1, 2, 3])
arr.shape                       # Array dimensions
arr.dtype                       # Data type
np.mean(arr), np.std(arr)      # Statistics
np.unique(arr)                  # Unique values

# Plotting basics
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('X label')
plt.ylabel('Y label')
plt.title('Title')
plt.legend()
plt.show()
```

## 🧠 ML/AI Quick Patterns

```python
# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# PyTorch model template
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training loop template
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# W&B logging
import wandb
wandb.init(project="my-project", name="experiment-1")
wandb.log({"loss": loss.item(), "accuracy": acc})
```

## 🐛 Troubleshooting

```bash
# Python version check
python --version
which python

# Pip issues
pip --version
pip cache purge

# Check if package installed
python -c "import numpy; print(numpy.__version__)"

# Virtual environment active?
echo $VIRTUAL_ENV              # Should show path to .venv

# Reinstall package
pip uninstall package-name
pip install package-name

# Jupyter kernel issues
python -m ipykernel install --user --name=ai-ml-env
# Then select kernel in Jupyter: Kernel > Change Kernel

# Clear Jupyter outputs (large notebooks)
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

## 🌐 Environment Variables

```bash
# Create .env file
echo "OPENAI_API_KEY=sk-..." >> .env
echo "WANDB_API_KEY=..." >> .env

# Use in Python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

## 🔥 Git Commands

```bash
# Daily workflow
git status                      # See what changed
git add .                       # Stage all changes
git commit -m "Week X: description"
git push origin main

# Check history
git log --oneline               # Compact history
git diff                        # See unstaged changes

# Create weekly branch (optional)
git checkout -b week-01
git push -u origin week-01
```

## 📁 Project Structure

```
journal/weeks/WEEK_XX/
├── README.md              # Week notes and learnings
├── *.ipynb                # Jupyter notebooks
├── *.py                   # Python utilities
└── reading_list.md        # Papers and resources
```

## 📅 Weekly Workflow

**Monday** - Theory (60 min)
```bash
# Read concepts, watch videos
# Take notes in README.md
```

**Tuesday** - Papers (60 min)
```bash
# Skim 2-3 papers
# Add to reading_list.md
```

**Wednesday** - Code (60 min)
```bash
jupyter lab
# Implement from scratch
# Create new .ipynb
```

**Thursday** - Experiment (60 min)
```bash
# Try variations
# Visualize results
# Document findings
```

**Friday** - Deliverable (60 min)
```bash
# Clean up notebook
# Write summary in README.md
# Commit and push
git add .
git commit -m "Week X complete"
git push
```

## 💡 Pro Tips

1. **Always activate venv**: Check with `echo $VIRTUAL_ENV`
2. **Save often**: Jupyter auto-saves, but commit to git frequently
3. **Document as you go**: Notes in markdown cells or README.md
4. **Experiment freely**: Copy notebooks to try variations
5. **Keep code simple**: Clarity > cleverness when learning
6. **Visualize everything**: Plots help build intuition
7. **Read the docs**: Better than guessing
8. **Ask why**: Don't just copy code, understand it

## 🎯 Learning Resources

**Documentation**
- Python: https://docs.python.org/3/
- NumPy: https://numpy.org/doc/
- Pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/
- scikit-learn: https://scikit-learn.org/stable/
- PyTorch: https://pytorch.org/docs/
- Jupyter: https://jupyterlab.readthedocs.io/

**In This Repo**
- `README.md` - Full 52-week plan
- `courses/` - Course references
- `certs/` - Certification guides
- `SETUP.md` - Detailed setup help

## ⌨️ Essential Shortcuts

**Terminal**
- `Ctrl+C` - Stop running process
- `Ctrl+D` - Exit Python/IPython
- `↑` / `↓` - Command history
- `Ctrl+R` - Search history
- `Ctrl+L` - Clear screen

**Jupyter Lab**
- `Shift+Enter` - Run cell
- `Esc` then `A` - Insert above
- `Esc` then `B` - Insert below
- `Esc` then `D D` - Delete cell
- `Ctrl+Shift+C` - Command palette

---

**Print this page** and keep it nearby for quick reference!

*Version 1.0 - Learning-focused*
