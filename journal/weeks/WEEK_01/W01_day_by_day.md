# Week 1 — PCA via SVD (Day-by-Day: Read / Watch / Do)

> Put each day under its own markdown header inside one notebook (e.g., `W01_PCA.ipynb`). Run the **setup cell once** at the top.

### One-time setup cell (run at the top of your notebook)
```python
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
%matplotlib inline
rng = np.random.default_rng(0)
```

---

## Day 1 — Linear algebra quick pass (vectors, norms, orthogonality, projections)

**Read (≈20m)**
- Vectors & inner product: \(x^\top y\); angle via \(\cos\theta=\frac{x^\top y}{\|x\|\,\|y\|}\).
- Norms: \(\ell_1,\ \ell_2,\ \ell_\infty\); triangle inequality; unit vectors.
- Orthonormality & orthogonal matrices: \(Q^\top Q=I\).
- Shapes & outer product: if \(A\in\mathbb{R}^{n\times d}\), \(B\in\mathbb{R}^{d\times k}\Rightarrow AB\in\mathbb{R}^{n\times k}\). Rank-1 outer product \(uv^\top\).
- Skim a reference (e.g., “Matrix Cookbook”) on **SVD** (definition/properties) and **pseudo-inverse**.

**Watch (optional, 5–10m)**
- 3Blue1Brown “Essence of Linear Algebra – SVD” (for geometric intuition).

**Do (≈30–35m)** — _add each block as its own cell_
```python
# Norms & triangle inequality
X = rng.normal(size=(3, 4))
for i, x in enumerate(X):
    l1, l2, linf = np.sum(np.abs(x)), np.linalg.norm(x), np.max(np.abs(x))
    print(f"v{i}: ||.||1={l1:.3f}, ||.||2={l2:.3f}, ||.||inf={linf:.3f}")
x, y = X[0], X[1]
print("triangle inequality:", np.linalg.norm(x+y) <= np.linalg.norm(x)+np.linalg.norm(y))
```
```python
# Projection onto a line P = (u u^T) / (u^T u)
u = rng.normal(size=(4,))
P = np.outer(u, u) / (u @ u)
print("symmetric:", np.allclose(P, P.T))
print("idempotent (P^2=P):", np.allclose(P @ P, P))

x = rng.normal(size=(4,))
x_proj = P @ x
# inner product with a component orthogonal to u should be ~0
orth = x - (x @ u)/(u @ u) * u
print("x_proj ⟂ orthogonal part:", np.isclose(x_proj @ orth, 0, atol=1e-8))
```
```python
# (Optional) Gram–Schmidt
def gram_schmidt(A, eps=1e-12):
    Q = []
    for v in A.T:
        w = v.astype(float)
        for q in Q:
            w -= q * (q @ w)
        n = np.linalg.norm(w)
        if n > eps:
            Q.append(w/n)
    return np.stack(Q, axis=1)

A = rng.normal(size=(4, 2))
Q = gram_schmidt(A)
print("Q^T Q ≈ I:", np.allclose(Q.T @ Q, np.eye(Q.shape[1])))
```

**Save/Note:** 3–5 bullet takeaways + a screenshot showing \(P^\top=P\) and \(P^2=P\).

---

## Day 2 — SVD ↔ PCA link (why \(V\) are principal directions; EVR from \(\sigma\))

**Read (≈25m)**
- Center first: \(X_c = X - \mathbf{1}\mu^\top\).
- SVD on centered data: \(X_c = U\Sigma V^\top\).
- Covariance: \(\frac{1}{n-1}X_c^\top X_c = V\,\frac{\Sigma^2}{n-1}\,V^\top\).
  - ⇒ Right singular vectors \(V\) are PCA directions.
  - ⇒ Eigenvalues of covariance are \(\sigma_i^2/(n-1)\).
- Explained variance ratio (EVR): \(\text{EVR}_i=\frac{\sigma_i^2}{\sum_j \sigma_j^2}\).

**Watch (optional, 5m)**
- Any short “PCA intuition via SVD” explainer.

**Do (≈30–35m)**
```python
# EVR from SVD
X = rng.normal(size=(6, 3))
mu = X.mean(axis=0, keepdims=True)
Xc = X - mu
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
V = Vt.T

evr = (s**2) / (s**2).sum()
print("σ:", s)
print("EVR:", evr, "sum:", evr.sum())
```
```python
# Covariance eigenvalues ≙ Σ^2/(n-1)
Cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
w, _ = np.linalg.eigh(Cov)   # ascending
print("eigvals(Cov) desc:", w[::-1])
print("Σ^2/(n-1):       ", (s**2)/(Xc.shape[0]-1))
```

**Save/Note:** write the EVR derivation in your words and confirm numerically that sums ≈ 1.

---

## Day 3 — Implement PCA via SVD (center → SVD → project → reconstruct)

**Read (≈10–15m)**
- Pipeline: center → economy SVD → pick \(k\) → scores \(Z=X_cV_k\) → reconstruct \(\hat X=ZV_k^\top+\mu\).
- Economy SVD is stable and avoids explicitly forming \(X_c^\top X_c\).

**Do (≈40–45m)**
```python
# Reusable helpers
def pca_fit_svd(X):
    X = np.asarray(X)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    evr = (s**2) / (s**2).sum()
    return mu, V, s, evr

def pca_transform(X, mu, V, k):
    return (X - mu) @ V[:, :k]

def pca_inverse_transform(Z, mu, V, k):
    return Z @ V[:, :k].T + mu
```
```python
# 2D demo + rank-1 reconstruction
N = 500
stds = np.array([3.0, 0.7])
X2 = rng.normal(size=(N,2)) * stds
theta = np.deg2rad(35)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
X2 = X2 @ R.T

mu2, V2, s2, evr2 = pca_fit_svd(X2)
Z1 = pca_transform(X2, mu2, V2, 1)
X2_hat1 = pca_inverse_transform(Z1, mu2, V2, 1)
print("EVR:", evr2, "sum:", evr2.sum())
print("MSE k=1:", np.mean((X2 - X2_hat1)**2))

plt.figure(figsize=(5,5))
plt.scatter(X2[:,0], X2[:,1], s=8, alpha=0.35, label='data')
plt.scatter(X2_hat1[:,0], X2_hat1[:,1], s=8, alpha=0.35, label='recon (k=1)')
plt.legend(); plt.axis('equal'); plt.title('Rank-1 PCA reconstruction'); plt.show()
```

**Save/Note:** EVR printout, MSE for \(k=1\). (Toggle to \(k=2\) to see error ≈ 0.)

---

## Day 4 — Evaluation & choosing \(k\) (scree, cumulative EV, reconstruction curve)

**Read (≈15m)**
- Scree & cumulative EV; pick \(k\) at 95% EV (or domain-driven).
- Eckart–Young idea: rank-\(k\) SVD is best low-rank approximation in Frobenius/2-norm.
- Data hygiene: **fit PCA on train only**, apply transform to val/test.

**Do (≈40–45m)**
```python
# Helper + 3D example
def choose_k_by_threshold(evr, threshold=0.95):
    c = np.cumsum(evr)
    return int(np.searchsorted(c, threshold) + 1), c

A = np.array([[1.0, 0.8, 0.6],
              [0.8, 2.0, 1.1],
              [0.6, 1.1, 1.5]])
L = np.linalg.cholesky(A)
X3 = rng.normal(size=(600,3)) @ L.T

mu3, V3, s3, evr3 = pca_fit_svd(X3)
k95, cum = choose_k_by_threshold(evr3, 0.95)
print("k for 95% EV:", k95)
```
```python
# Cumulative EV plot
plt.figure(figsize=(5,4))
plt.plot(np.arange(1, len(evr3)+1), cum, marker='o')
plt.axhline(0.95, linestyle='--')
plt.xlabel('k'); plt.ylabel('cumulative EV'); plt.title('Cumulative explained variance')
plt.show()
```
```python
# Reconstruction error vs k
mses = []
for k in range(1, X3.shape[1]+1):
    Zk = pca_transform(X3, mu3, V3, k)
    Xhat = pca_inverse_transform(Zk, mu3, V3, k)
    mses.append(np.mean((X3 - Xhat)**2))

plt.figure(figsize=(5,4))
plt.plot(range(1, len(mses)+1), mses, marker='o')
plt.xlabel('k'); plt.ylabel('MSE'); plt.title('Reconstruction error vs k')
plt.show()
```

**Save/Note:** your chosen \(k_{0.95}\), both plots, and a one-liner on leakage prevention.

---

## Day 5 — Whitening & eigen route (compare methods; standardization policy)

**Read (≈20m)**
- PCA whitening: divide scores by \(\sigma_{1:k}\) (or \(\sqrt{\lambda_i}\)) → \(\mathrm{Cov} \approx I\).
- Eigen route vs SVD route: agree on centered data (up to sign/order).
- When to standardize: features with different units/scales or dominated variance; compute mean/std on **train only**.

**Do (≈35–40m)**
```python
# Whitening check
def pca_whiten(Z, s, k, eps=1e-6):
    return Z / (s[:k] + eps)

k = 2
Z = pca_transform(X3, mu3, V3, k)
Zw = pca_whiten(Z, s3, k)
print("Cov(Z):\n", np.cov(Z.T))
print("Cov(Z whitened):\n", np.cov(Zw.T))
```
```python
# Covariance eigen-decomposition PCA (compare to SVD)
def pca_fit_eig(X):
    X = np.asarray(X)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / (X.shape[0] - 1)
    w, V = np.linalg.eigh(C)           # ascending
    idx = np.argsort(w)[::-1]          # descending
    w, V = w[idx], V[:, idx]
    s = np.sqrt(w * (X.shape[0] - 1))  # singular values
    evr = w / w.sum()
    return mu, V, s, evr

mu_e, V_e, s_e, evr_e = pca_fit_eig(X3)
print("EVR match:", np.allclose(evr3, evr_e))
print("Principal directions align (abs diag V^T V_e):", np.abs(np.diag(V3.T @ V_e)))
```

**Save/Note:** covariance numbers before/after whitening, EVR match True/False, and your **standardization rule of thumb**.

---

## (Optional) Stretch for later
- Randomized SVD; Incremental PCA (streaming).
- Tall/skinny scaling tips; avoid explicitly forming \(X^\top X\) when \(d\) is large.
- Common pitfalls: not centering; fitting on full data (leakage); over-interpreting PCs causally.

---

### End-of-Week Checklist
- [ ] Day 1: projection matrix verified (\(P^\top=P\), \(P^2=P\))  
- [ ] Day 2: EVR derivation + numeric confirmation  
- [ ] Day 3: PCA helpers + 2D rank-1 reconstruction plot  
- [ ] Day 4: cumulative EV + chosen \(k\) and MSE curve  
- [ ] Day 5: whitening covariance ≈ identity + eigen/SVD agreement + standardization policy
