# Reading List & Micro-Schedule (5 × ~60 min)

## Day 1 — Linear algebra quick pass
- Vectors, norms, dot products, orthogonality
- Matrix multiplication shapes; projections
- **Read:** The Matrix Cookbook (sections on SVD & properties) — skim
- **Watch (optional):** 3Blue1Brown: Essence of Linear Algebra (SVD)

## Day 2 — SVD & PCA link
- Why SVD on centered X gives PCA directions (V)
- Explained variance from singular values
- **Do:** Derive EVR = s²/(n−1)/∑s² and write it in your journal

## Day 3 — Implement PCA via SVD
- Center → SVD (economy) → project → reconstruct
- **Do:** Implement from scratch (use `pca_utils.py` pattern)

## Day 4 — Evaluation & choosing k
- Scree + cumulative explained variance
- Reconstruction error vs k
- **Do:** Plot both; decide k for 95% on a dataset

## Day 5 — Whitening & eigen route
- PCA whitening and its effect on covariance
- Compare SVD-PCA vs covariance eigen-decomposition
- **Do:** Summarize when to standardize features

**Stretch (for later):** randomized SVD, incremental PCA, scaling to tall/skinny matrices; pitfalls (not centering, mixing train/test stats).

