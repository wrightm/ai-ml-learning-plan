# Linear Algebra 101 for ML

Here’s a crisp “Linear Algebra 101 for ML” — what each idea is, why it matters, the minimum you should remember, and tiny checks you can do in NumPy.

---

## 1) Vectors, Norms, and Inner Products

**What:** Vectors are points/arrows in (\mathbb{R}^d).
**Key ops:** length (|x|*2=\sqrt{x^\top x}); dot product (x^\top y); angle via (\cos\theta=\frac{x^\top y}{|x|,|y|}).
**Why ML:** similarity (cosine), embedding geometry, gradients.
**Remember:** (\ell_1) promotes sparsity; (\ell_2) is smooth; (\ell*\infty) = max component.

```python
# NumPy check
x,y = rng.normal(size=5), rng.normal(size=5)
cos = (x@y)/(np.linalg.norm(x)*np.linalg.norm(y))
```

---

## 2) Matrices & Linear Maps

**What:** A matrix (A) represents a linear map (x\mapsto Ax). Shapes: ((n\times d)(d\times k)=(n\times k)).
**Why ML:** data matrix (X\in\mathbb{R}^{n\times d}) (rows=samples, cols=features); predictions (Xw).
**Remember:** Column space (span of columns); rank = dimension of column space.
**Pitfall:** Row vs column orientation changes shapes.

---

## 3) Orthogonality & Projections

**What:** (x\perp y\iff x^\top y=0). Projection onto direction (u): (P=\dfrac{uu^\top}{u^\top u}).
**Why ML:** least squares is an orthogonal projection of (y) onto span of (X)’s columns.
**Remember:** Projection matrices are symmetric & idempotent: (P^\top=P), (P^2=P).

```python
# NumPy check
u = rng.normal(size=4)
P = np.outer(u,u)/(u@u)
np.allclose(P,P.T), np.allclose(P@P,P)
```

---

## 4) Least Squares (Linear Regression)

**Problem:** (\min_w |Xw - y|_2^2).
**Solution:** normal equations (X^\top X,w = X^\top y); if full column rank, (w=(X^\top X)^{-1}X^\top y).
**Why ML:** baseline predictor; building block for GLMs.
**Numerics:** prefer `np.linalg.lstsq(X,y)` or QR/SVD over explicit inversion.

---

## 5) Eigenvalues/Eigenvectors (Symmetric Case)

**What:** (A v=\lambda v). For symmetric (A), real eigenvalues, orthonormal eigenvectors.
**Why ML:** covariance matrices, Laplacians, power iteration, spectral clustering.
**Remember:** For PSD (A\succeq 0), all (\lambda_i\ge 0).

```python
# NumPy check
A = rng.normal(size=(4,4)); A = (A+A.T)/2
w,V = np.linalg.eigh(A)  # eigenvalues (w) & eigenvectors (V)
```

---

## 6) SVD (The Workhorse)

**What:** (A=U\Sigma V^\top) with orthonormal (U,V), singular values (\sigma_i\ge 0).
**Why ML:** low-rank structure, PCA, pseudo-inverse, denoising, recommender systems.
**Remember:** best rank-(k) approximation: keep top (k) singular triplets.
**Pseudoinverse:** (A^+=V\Sigma^+U^\top) (invert nonzero (\sigma_i)).

```python
# NumPy check
U,S,Vt = np.linalg.svd(A, full_matrices=False)
k=2
Ak = (U[:,:k]*S[:k]) @ Vt[:k,:]
```

---

## 7) PCA in One Line

**What:** PCA finds directions of maximal variance.
**Two views:**

* SVD of centered data (X_c=U\Sigma V^\top): PCs = columns of (V).
* Eigen of covariance (C=\frac{1}{n-1}X_c^\top X_c = V \Lambda V^\top) with (\Lambda=\frac{\Sigma^2}{n-1}).
  **Why ML:** dimensionality reduction, visualization, denoising, whitening.
  **Remember:** explained variance ratio (=\dfrac{\sigma_i^2/(n-1)}{\sum_j \sigma_j^2/(n-1)}).

---

## 8) Positive-(Semi)Definite (PD/PSD) Matrices

**What:** (A) is PSD if (x^\top Ax\ge 0) for all (x).
**Why ML:** covariance matrices, kernels/Gram matrices (K=XX^\top), Gaussians.
**Remember:** PSD (\Rightarrow) eigenvalues (\ge 0); Cholesky exists for PD.

---

## 9) Conditioning & Numerical Stability

**What:** Condition number (\kappa(A)=\sigma_{\max}/\sigma_{\min}).
**Why ML:** large (\kappa) → unstable solutions, exploding parameter variance.
**Fixes:** feature scaling/standardization, regularization (ridge), use SVD/QR.

---

## 10) Regularization (Ridge & Lasso)

* **Ridge:** (\min_w |Xw-y|^2 + \lambda|w|_2^2) → (w=(X^\top X+\lambda I)^{-1}X^\top y).
* **Lasso:** (\min_w |Xw-y|^2 + \lambda|w|_1) (promotes sparsity; solved by convex opt).
  **Why ML:** combats overfitting, improves conditioning.

---

## 11) Gradients, Jacobians, Matrix Calculus (Just Enough)

* (\nabla_x (a^\top x)=a)
* (\nabla_x \tfrac12|Ax-b|_2^2 = A^\top (Ax-b))
* If (f(x)=\tfrac12 x^\top Q x + c^\top x) with symmetric (Q): (\nabla f=Qx+c).
  **Why ML:** derive updates, verify autograd, design custom layers/losses.

---

## 12) Orthonormal Bases, QR, Gram–Schmidt

**What:** (A=QR) with (Q^\top Q=I) and upper triangular (R).
**Why ML:** stable way to solve least squares; Gram–Schmidt builds orthonormal bases.

```python
# NumPy check
Q,R = np.linalg.qr(X, mode='reduced')
np.allclose(Q.T@Q, np.eye(Q.shape[1]))
```

---

## 13) Low-Rank Structure

**What:** Many real datasets are approximately low-rank.
**Why ML:** compression (PCA), fast models, collaborative filtering, attention approximations.

---

## 14) Kernels (Peek Ahead)

**What:** Replace dot products with (k(x,y)) (PSD kernels).
**Why ML:** kernel SVMs, GP regression.
**Remember:** Gram matrix (K_{ij}=k(x_i,x_j)) must be PSD.

---

## How These Show Up in Common ML Models

* **Linear/Logistic regression:** (X^\top X), conditioning, regularization.
* **PCA:** SVD of centered data, EVR for choosing (k).
* **Word/image embeddings:** cosine similarity, nearest neighbors in (\ell_2).
* **Attention (Transformers):** softmax((QK^\top/\sqrt{d})V) — dot products + matrix multiplies dominate cost.
* **Convolutions:** linear maps with structured Toeplitz/Circulant matrices (implemented efficiently via kernels/FFT).
* **Gaussian models:** PSD covariance, Cholesky for likelihood/inference.

---

## Minimal Formula Sheet (Memorize)

* (|x|_2=\sqrt{x^\top x}), (\cos\theta=\dfrac{x^\top y}{|x||y|}), (x\perp y \iff x^\top y=0)
* Projection onto (u): (P=\dfrac{uu^\top}{u^\top u}), (P^\top=P), (P^2=P)
* Least squares gradient: (\nabla_w \tfrac12|Xw-y|^2=X^\top(Xw-y))
* Ridge solution: (w=(X^\top X+\lambda I)^{-1}X^\top y)
* SVD: (A=U\Sigma V^\top), PCA: (V) = principal directions, EVR (\propto \sigma_i^2)
* Condition number: (\kappa=\sigma_{\max}/\sigma_{\min})

---

## Quick Practice Set (with Tiny NumPy Nudges)

1. **Angles & cosines:** hand-compute angle between (x=(2,1,0)) and (y=(1,2,2)).

   ```python
   x=np.array([2,1,0.]); y=np.array([1,2,2.])
   ```

2. **Projection:** project (x=(3,1,2)) onto (u=(1,1,1)); verify residual is orthogonal.

   ```python
   u=np.ones(3); x=np.array([3.,1,2])
   ```

3. **Least squares:** with
   (X=\begin{bmatrix}1&0\1&1\1&2\end{bmatrix}), (y=(1,2,2)), solve (w) via `lstsq` and via normal equations.

4. **Ridge vs OLS:** same (X,y) as (3), compare (w_{\text{OLS}}) to (w_{\text{ridge}}) for (\lambda=1).

   ```python
   lam=1.0; w_ridge = np.linalg.solve(X.T@X+lam*np.eye(2), X.T@y)
   ```

5. **SVD truncation:** make (A\in\mathbb{R}^{10\times 10}) rank-2 by construction (outer products), add small noise, recover rank-2 approx with top-2 singular values.

6. **PCA equivalence:** center (X); verify: right singular vectors (V) equal eigenvectors of (X^\top X) and eigenvalues match (\sigma_i^2).

7. **QR solve:** solve (\min_w|Xw-y|) using QR: (w=R^{-1}Q^\top y). Compare with `lstsq`.

8. **PSD check:** sample (B) and set (A=B^\top B). Show (x^\top Ax\ge 0) for random (x).

9. **Conditioning:** scale one feature of (X) by (10^6); observe condition number and OLS instability.

10. **Cosine vs Euclidean:** create three 2D points; rank neighbors of a query under cosine vs Euclidean after (z)-score normalization; note differences.

---

## Common Pitfalls (and Fixes)

* **Feature scale mismatch:** always standardize/normalize before regression/PCA.
* **Explicit inverses:** avoid `inv`; use `solve`, QR, or SVD.
* **Rank deficiency:** if (X^\top X) is ill-conditioned, use ridge or SVD-based solvers.
* **Orientation errors:** be consistent (rows=samples, cols=features).
