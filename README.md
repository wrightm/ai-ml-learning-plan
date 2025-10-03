# AI & ML Learning – Agentic AI + Recommenders (1 hour/day for 52 weeks)

<!--PROGRESS_BADGE_START-->
![Progress](https://img.shields.io/badge/Progress-0%25-brightgreen)
<!--PROGRESS_BADGE_END-->

## 💪 One Hour a Day - Arnold Schwarzenegger

> *"The worst thing I can be is the same as everybody else. I hate that."* - Arnold Schwarzenegger

Click to watch Arnold's motivation on the power of one hour a day:

[![Arnold Schwarzenegger - One Hour a Day](https://img.youtube.com/vi/mPo-YY5kMOA/maxresdefault.jpg)](https://www.youtube.com/watch?v=mPo-YY5kMOA "One Hour a Day - Arnold Schwarzenegger")

**The message**: One hour a day, dedicated and focused, is all it takes to transform yourself. No excuses.

---

## The Plan

A ruthless, focused plan for a senior staff engineer (math/physics background, a bit rusty) to become **expert in Agentic AI** and **Recommendation Systems** in one year with ~**1 hour/day**.

> **Cadence**: Mon–Fri → ~60m/day (theory, paper skim, coding, experiment, weekly deliverable). Use `journal/weeks/WEEK_XX/` to log progress.

## Repo layout
```
.
├─ README.md                    # This file - your learning plan
├─ SETUP.md                     # Detailed setup guide
├─ QUICK_REFERENCE.md           # Command cheat sheet
├─ requirements.txt             # Python dependencies
├─ Makefile                     # Convenient commands
├─ journal/weeks/               # Your weekly work goes here
│  └─ WEEK_XX/
│     ├─ README.md              # Week notes
│     ├─ *.ipynb                # Jupyter notebooks
│     └─ *.py                   # Python utilities
├─ courses/                     # Course references
├─ certs/                       # Certification guides
└─ scripts/
   └─ update_badge.py           # Updates progress badge
```

> Track your progress by checking boxes below as you complete each week.

---

## Getting started

### 1. Python Environment Setup
This project requires **Python 3.11+**. Quick start:

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

> See [SETUP.md](SETUP.md) for detailed installation instructions and troubleshooting.

### 2. Start Learning
```bash
# Launch Jupyter Lab
jupyter lab
# OR use Makefile
make lab
```

Your browser will open. Navigate to `journal/weeks/WEEK_01/W01_day_by_day.ipynb` to start!

### 3. Weekly Workflow
- **Mark progress**: Check boxes below as you complete weeks (`- [ ]` → `- [x]`)
- **Take notes**: Document learnings in `journal/weeks/WEEK_XX/README.md`
- **Code daily**: Aim for ~60 minutes, Monday–Friday
- **Commit work**: Push to GitHub to track your progress

> **Optional**: Set up [GitHub Actions](.github/) to auto-update the progress badge above.

---

## 52‑Week Plan at a Glance
**Weekly rhythm**
- **Mon:** theory (videos/reading) ~60m  
- **Tue:** skim & notes (papers/blogs) ~60m  
- **Wed:** code from scratch (notebooks/scripts) ~60m  
- **Thu:** extend/experiment ~60m  
- **Fri:** mini-deliverable (writeup, PR, or result) ~60m  

**Core Stack**
- **Foundation**: Python 3.11+, NumPy, Pandas, Matplotlib, scikit-learn, Jupyter
- **Deep Learning**: PyTorch, Lightning
- **Experimentation**: Weights & Biases
- **Recommender Systems**: implicit, LightFM, FAISS
- **LLM & Agents**: LangChain, OpenAI API, instructor
- **Later phases**: FastAPI (serving), SQLAlchemy (data)

---

## Phase 1 — Math Reboot & ML Foundations (Weeks 1–12)
- [ ] **W1 — Vectors & Matrices refresher** → PCA via SVD notebook
- [ ] **W2 — Probability & Stats I** → CLT simulation & sampling demos
- [ ] **W3 — Probability & Stats II** → A/B test notebook + power analysis
- [ ] **W4 — Optimization** → Logistic regression from scratch (GD)
- [ ] **W5 — ML Experiment Tracking** → W&B setup, logging experiments, comparing runs
- [ ] **W6 — Supervised I** → Linear/regularized models baseline
- [ ] **W7 — Supervised II** → Trees/GBM comparison + SHAP
- [ ] **W8 — Unsupervised** → Clustering + visualization
- [ ] **W9 — Evaluation** → Metrics utils (ROC/PR, calibration, ranking)
- [ ] **W10 — Data/Serving** → FastAPI model service (artifact‑pinned)
- [ ] **W11 — Experimentation** → Ablation template + example
- [ ] **W12 — Checkpoint 1** → Rebuild LR + GBM; 2–3p memo

## Phase 2 — Deep Learning & Representations (Weeks 13–20)
- [ ] **W13 — PyTorch fundamentals** → MLP on MNIST/CIFAR with clean loops
- [ ] **W14 — Optimization in practice** → Recipe card reaching target accuracy
- [ ] **W15 — Embeddings & metric learning** → Contrastive toy + retrieval
- [ ] **W16 — Sequence models** → Next‑item predictor (synthetic data)
- [ ] **W17 — Transformers** → Fine‑tune small LM; eval perplexity
- [ ] **W18 — Retrieval & Vector Search** → Retrieval API over embeddings
- [ ] **W19 — Serving & Profiling** → Latency/QPS benchmarks; FastAPI endpoint
- [ ] **W20 — Checkpoint 2** → Write “What I know about embeddings & sequences”

## Phase 3 — Recommendation Systems (Weeks 21–32)
- [ ] **W21 — Framing & Metrics** → Ranking harness (AUC/MAP/NDCG)
- [ ] **W22 — Nearest‑neighbor CF** → Top‑K baseline + caching
- [ ] **W23 — Matrix Factorization** → MF (ALS/SGD) + BPR; compare to W22
- [ ] **W24 — FMs & Wide‑and‑Deep** → Sparse features vs MF
- [ ] **W25 — Two‑Tower Retrieval** → Neg sampling; Recall@K lift
- [ ] **W26 — Re‑ranking** → GBM/transformer re‑ranker; NDCG lift
- [ ] **W27 — Sequential Recsys** → SASRec/GRU4Rec ablations
- [ ] **W28 — Explore/Exploit** → Bandits; offline IPS/DR evaluation
- [ ] **W29 — Causality/Uplift (intro)** → Toy uplift & product notes
- [ ] **W30 — Practical Evaluation** → Guardrails; diversity/serendipity metrics
- [ ] **W31 — Cold‑start & Taxonomy** → Content features & hierarchy
- [ ] **W32 — Checkpoint 3 (Mini‑Capstone #1)** → Retrieval→re‑rank pipeline + API

## Phase 4 — Agentic AI (Weeks 33–42)
- [ ] **W33 — Agent architectures** → Compare planner/executor, debate, graphs
- [ ] **W34 — Tool use & function calling** → 3 tools + schema validation
- [ ] **W35 — Multi‑agent patterns** → Supervisor/worker demo
- [ ] **W36 — Memory & reflection** → Persistent memory; staleness tests
- [ ] **W37 — Planning** → Multi‑step task planner (simulated web)
- [ ] **W38 — RAG done right** → Evaluation (faithfulness, relevancy)
- [ ] **W39 — Agent evaluation** → Trace KPIs; harness
- [ ] **W40 — Safety & reliability** → Guardrails; PII; timeouts/circuit breakers
- [ ] **W41 — Local vs hosted** → Adapter to swap LLM backends
- [ ] **W42 — Checkpoint 4 (Mini‑Capstone #2)** → Task‑oriented agent demo

## Phase 5 — Integrating Agents + Recsys (Weeks 43–48)
- [ ] **W43 — Recommendation agent** → Offline evals + config PRs
- [ ] **W44 — Conversational recsys** → Preference updates & re‑ranking
- [ ] **W45 — Exploration agent** → Auto‑tagging + A/B proposals
- [ ] **W46 — Bandit + agent loop** → Simulated regret reduction
- [ ] **W47 — MLOps hardening** → Feature store, drift, CI/CD, canary
- [ ] **W48 — Checkpoint 5** → Operating Manual for your platform

## Phase 6 — Capstone & Portfolio (Weeks 49–52)
- [ ] **W49–W51 — Capstone** → Pick 1 (or both): Recsys or Agentic platform
- [ ] **W52 — Polish** → Slides, blog(s), demo video, portfolio

---

## Journal Structure
Each week, work in `journal/weeks/WEEK_XX/`:
```md
WEEK_XX/
├── README.md           # Week notes and learnings
├── *.ipynb             # Jupyter notebooks with implementations
├── *.py                # Reusable Python utilities
└── reading_list.md     # Papers and resources
```

**Example Week Notes** (`README.md`):
```md
# Week 01 — Vectors & Matrices
**Goals:** PCA via SVD; refresh linear algebra fundamentals  
**What I Learned:** SVD decomposition, explained variance, whitening  
**Code:** W01_day_by_day.ipynb  
**Key Insights:** PCA is just eigendecomposition of covariance...  
**Next Week:** Probability & stats refresher
```

## Tips for Success
- **Consistency > intensity**: 1 hour/day beats 7 hours on Sunday
- **Code everything from scratch**: Understanding > efficiency when learning
- **Visualize concepts**: Plots build intuition
- **Document as you go**: Future you will be grateful
- **Review previous weeks**: Spaced repetition aids retention
- **Don't skip fundamentals**: Strong foundations enable advanced work

---

### Credits / Reading List (sprinkle across the year)
- Math: *Matrix Cookbook*, Murphy’s *ML: A Probabilistic Perspective* (selected), Boyd & Vandenberghe (convex chapters).
- Recsys: *Recommender Systems Handbook* (chapters), Rendle (BPR), He et al. (Neural CF), Hidasi et al. (GRU4Rec), Kang & McAuley (SASRec), Covington et al. (YouTube).
- Agents & RAG: LangGraph docs, function-calling & structured outputs, multi-agent surveys, retrieval evaluation guides.
- MLOps: Chip Huyen, blog posts on drift/lineage/feature stores.

---

Happy building—check a box each week and ship something small every Friday.
