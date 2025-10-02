# AI & ML Learning – Agentic AI + Recommenders (1 hour/day for 52 weeks)

<!--PROGRESS_BADGE_START-->
![Progress](https://img.shields.io/badge/Progress-0%25-brightgreen)
<!--PROGRESS_BADGE_END-->

A ruthless, focused plan for a senior staff engineer (math/physics background, a bit rusty) to become **expert in Agentic AI** and **Recommendation Systems** in one year with ~**1 hour/day**.

> **Cadence**: Mon–Fri → ~60m/day (theory, paper skim, coding, experiment, weekly deliverable). Use `journal/Week_XX.md` to log progress.

## Repo layout
```
.
├─ README.md
├─ journal/                     # Your weekly notes live here
├─ projects/                    # Capstones & mini-capstones
├─ scripts/
│  └─ update_badge.py           # Auto-updates the progress badge from README checkboxes
└─ .github/
   ├─ workflows/
   │  ├─ update-progress.yml    # Updates badge daily and on pushes
   │  └─ weekly-issue.yml       # Opens a weekly check‑in issue every Monday
   └─ ISSUE_TEMPLATE/
      └─ weekly_progress.md     # Template for weekly issues
```
> After your **first push**, GitHub Actions will run and set your progress badge. Edit the checkboxes below as you complete weeks.

---

## Getting started
1. **Create a new GitHub repo** and upload all files from this folder.
2. Open the **Actions** tab and enable workflows (if prompted).
3. (Optional) Adjust the schedule time in `.github/workflows/weekly-issue.yml` to your preference.
4. Each Monday, a **Weekly Check‑in** issue is created. Use it to plan and reflect.
5. Mark each completed week below by changing `- [ ]` → `- [x]`. The badge updates automatically.

---

## 52‑Week Plan at a Glance
**Weekly rhythm**
- **Mon:** theory (videos/reading) ~60m  
- **Tue:** skim & notes (papers/blogs) ~60m  
- **Wed:** code from scratch (notebooks/scripts) ~60m  
- **Thu:** extend/experiment ~60m  
- **Fri:** mini-deliverable (writeup, PR, or result) ~60m  

**Core Stack**
- Python 3.11+, PyTorch, numpy/pandas, scikit‑learn, matplotlib, Jupyter
- Lightning/Fabric (optional), Weights & Biases (tracking), Poetry/uv (env), pytest, ruff/black
- FastAPI, Docker, SQLite/Postgres
- Vector Search: FAISS/HNSW; Pinecone/Weaviate/Chroma (optional)
- Recsys libs later: implicit, LightFM, Recbole/Merlin; banditpylib
- Agent libs: LangGraph, pydantic/instructor, vLLM/Ollama (optional), LLM client(s)

---

## Phase 1 — Math Reboot & ML Foundations (Weeks 1–12)
- [ ] **W1 — Vectors & Matrices refresher** → PCA via SVD notebook
- [ ] **W2 — Probability & Stats I** → CLT simulation & sampling demos
- [ ] **W3 — Probability & Stats II** → A/B test notebook + power analysis
- [ ] **W4 — Optimization** → Logistic regression from scratch (GD)
- [ ] **W5 — ML Project Template** → CI, tests, pre‑commit, W&B
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

## Journal
Create weekly files like `journal/Week_01.md`:
```md
# Week 01 — Vectors & Matrices
**Goals:** PCA via SVD; refresh norms/projections  
**Notes:** …  
**Code/Experiments:** links  
**Results:** screenshot/metrics  
**Next:** …
```

## Automation & Badges
- The **progress badge** updates automatically by counting completed `- [x]` checkboxes in the Week lists above. It runs **daily** and on **every push**.
- A **Weekly Check‑in** issue is created every **Monday** (UTC) with planning prompts. Adjust the schedule in the workflow if you prefer a different time.

## Optional: Email notifications
GitHub won’t email arbitrary addresses from Actions without a provider. To get weekly emails to yourself, add a mail action (e.g., SendGrid) and set secrets `SENDGRID_API_KEY` and `TO_EMAIL`. See comments inside `weekly-issue.yml`.

---

### Credits / Reading List (sprinkle across the year)
- Math: *Matrix Cookbook*, Murphy’s *ML: A Probabilistic Perspective* (selected), Boyd & Vandenberghe (convex chapters).
- Recsys: *Recommender Systems Handbook* (chapters), Rendle (BPR), He et al. (Neural CF), Hidasi et al. (GRU4Rec), Kang & McAuley (SASRec), Covington et al. (YouTube).
- Agents & RAG: LangGraph docs, function-calling & structured outputs, multi-agent surveys, retrieval evaluation guides.
- MLOps: Chip Huyen, blog posts on drift/lineage/feature stores.

---

Happy building—check a box each week and ship something small every Friday.
