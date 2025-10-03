## Day 1 — vectors, norms, orthogonality, projections

**Read (≈20m)**
- Vectors & inner product: \(x^\top y\); angle via \(\cos\theta=\frac{x^\top y}{\|x\|\,\|y\|}\).
- Norms: \(\ell_1,\ \ell_2,\ \ell_\infty\); triangle inequality; unit vectors.
- Orthonormality & orthogonal matrices: \(Q^\top Q=I\).
- Shapes & outer product: if \(A\in\mathbb{R}^{n\times d}\), \(B\in\mathbb{R}^{d\times k}\Rightarrow AB\in\mathbb{R}^{n\times k}\). Rank-1 outer product \(uv^\top\).
- Skim a reference (e.g., “Matrix Cookbook”) on **SVD** (definition/properties) and **pseudo-inverse**.

**Watch (optional, 5–10m)**
- 3Blue1Brown “Essence of Linear Algebra – SVD” (for geometric intuition).



### Notes

## 1) Vectors, Norms, and Inner Products

**What:** Vectors are points/arrows in $\mathbb{R}^d$.  
**Key ops:** length $\|x\|_2=\sqrt{x^\top x}$; dot product $x^\top y$; angle via $\cos\theta=\frac{x^\top y}{\|x\|\|y\|}$.  
**Why ML:** similarity (cosine), embedding geometry, gradients.  
**Remember:** $\ell_1$ promotes sparsity; $\ell_2$ is smooth; $\ell_\infty$ = max component.

```python
# NumPy check
x,y = rng.normal(size=5), rng.normal(size=5)
cos = (x@y)/(np.linalg.norm(x)*np.linalg.norm(y))
```

# Dot Product & Norms — Cheat Sheet

## Dot Product

**Definition**

For $x=(x_1,\dots,x_n)$ and $y=(y_1,\dots,y_n)$,

$$
x\cdot y = x^\top y = \sum_{i=1}^n x_i y_i.
$$

**Geometric meaning**

$$
x\cdot y = \|x\| \|y\| \cos\theta
$$

where $\theta$ is the angle between $x$ and $y$.

* Large positive → similar direction (small $\theta$)
* Zero → **orthogonal** (right angle)
* Negative → opposite directions

**Useful properties**

* Commutative: $x\cdot y=y\cdot x$
* Linear in each argument: $(ax+by)\cdot z=a(x\cdot z)+b(y\cdot z)$
* Tied to length: $\|x\|_2=\sqrt{x\cdot x}$

**Worked examples**

*2D (with angle)*  
Let $x=(3,1)$, $y=(1,3)$.

* $x\cdot y = 3\cdot1 + 1\cdot3 = 6$
* $\|x\|=\sqrt{10}$, $\|y\|=\sqrt{10}$
* $\cos\theta = \frac{6}{\sqrt{10}\sqrt{10}} = 0.6 \Rightarrow \theta \approx 53.13^\circ$

*3D (with projection)*  
Let $x=(2,-1,3)$, $u=(1,1,1)$.

* $x\cdot u=4$, $u\cdot u=3$
* $\mathrm{proj}_u(x)=\dfrac{x\cdot u}{u\cdot u}u=\dfrac{4}{3}(1,1,1)$

**NumPy check**

```python
import numpy as np
x = np.array([3.,1.]); y = np.array([1.,3.])
dot = x @ y
cos = dot / (np.linalg.norm(x)*np.linalg.norm(y))
dot, cos
```

---

## Norms

**What is a norm?**  
A function $\|\cdot\|$ that measures vector "size," satisfying for all $x,y$ and scalars $\alpha$:

1. **Non-negativity & definiteness:** $\|x\|\ge 0$, and $\|x\|=0 \iff x=0$
2. **Absolute scalability:** $\|\alpha x\|=|\alpha|\|x\|$
3. **Triangle inequality:** $\|x+y\|\le \|x\|+\|y\|$

**Common norms (on $\mathbb{R}^d$)**

* **$\ell_2$ (Euclidean):** $\displaystyle \|x\|_2=\sqrt{\sum_i x_i^2}=\sqrt{x^\top x}$
* **$\ell_1$ (Manhattan):** $\displaystyle \|x\|_1=\sum_i |x_i|$
* **$\ell_\infty$ (Max):** $\displaystyle \|x\|_\infty=\max_i |x_i|$

**Geometry**  
Unit balls in $\mathbb{R}^2$:

* $\ell_2$: circle
* $\ell_1$: diamond
* $\ell_\infty$: square

## Why ML Cares

Norms as Distances & Regularizers — Ridge vs Lasso 

### Big picture (plain English)

* **Norms define distances:** from a norm $\|x\|$ you get a distance $d(x,y)=\|x-y\|$. Different norms ⇒ different geometry ⇒ different behavior in models.
* **Regularizers:** add a penalty to the training loss to discourage overly complex models:
  
  $$
  \text{Loss}(w)=\underbrace{\text{fit error}}_{\|Xw-y\|_2^2}+\lambda\cdot\underbrace{\text{regularizer}(w)}_{\text{a norm of }w}.
  $$

* **Ridge ($\ell_2$)**: $\|w\|_2^2$. Smoothly shrinks all weights; great for correlated features; rarely makes exact zeros.
* **Lasso ($\ell_1$)**: $\|w\|_1$. Introduces "corners" at zero; drives some weights **exactly** to 0 (sparse, interpretable).

## Geometric intuition (why their behavior differs)

* Ridge's $\ell_2$ unit ball is a **circle** (smooth). The optimum tends to touch smooth boundaries → **shrink** but not exactly zero.
* Lasso's $\ell_1$ unit ball is a **diamond** (sharp corners on axes). The optimum often lands on corners → **feature coefficients exactly zero**.

## Practical guidance

* **Standardize** $X$ (zero-mean, unit-variance) before ridge/lasso so the penalty treats all features comparably.
* Tune $\lambda$ with **cross-validation**.
* If features are highly correlated, lasso may arbitrarily pick one and drop the rest. **Elastic Net** blends both norms to mitigate this.

---

## Fully commented NumPy demo (Ridge vs Lasso on a tiny synthetic problem)

```python
import numpy as np

# -----------------------------
# 1) Make a small, realistic dataset
# -----------------------------

rng = np.random.default_rng(0)                        # set a random seed for reproducibility

n = 80                                               # number of samples (rows)
# Create two highly correlated features: x2 ≈ x1 + small noise
x1 = rng.normal(size=n)                              # feature 1: standard normal
x2 = x1 + 0.05 * rng.normal(size=n)                  # feature 2: almost the same as x1 (multicollinearity)
x3 = rng.normal(size=n)                              # feature 3: mostly irrelevant/noise

# Stack columns into the design matrix X (shape n x d where d=3)
X = np.column_stack([x1, x2, x3])                    # X has 3 features

# Standardize features so each column has mean 0 and std 1 (important for fair regularization)
X = (X - X.mean(axis=0)) / X.std(axis=0)             # z-score scaling per feature

# Define the "true" underlying weights. Here both x1 and x2 matter, x3 doesn't.
w_true = np.array([1.0, 1.0, 0.0])                   # ground-truth coefficients

# Generate the target with some noise: y = X w_true + ε
y = X @ w_true + 0.2 * rng.normal(size=n)            # supervised learning target vector


# -----------------------------
# 2) Ridge regression (closed form)
# -----------------------------

lam = 1.0                                            # ridge penalty strength λ (tune via CV in practice)
d = X.shape[1]                                       # number of features

# Closed-form ridge solution: w_ridge = (X^T X + λ I)^{-1} X^T y
# NOTE: Using 'solve' is numerically safer than explicit inverse.
XtX = X.T @ X                                        # Gram matrix (d x d)
Xty = X.T @ y                                        # feature-target correlations (d,)
w_ridge = np.linalg.solve(XtX + lam * np.eye(d), Xty)# solve linear system for w_ridge

# -----------------------------
# 3) Lasso regression (coordinate descent, simple illustrative version)
# -----------------------------

def soft_thresh(z, a):
    """
    Soft-thresholding operator:
      argmin_w (1/2)*(w - z)^2 + a*|w|  =>  sign(z) * max(|z| - a, 0)
    This is the proximal step that induces exact zeros for small |z|.
    """
    return np.sign(z) * np.maximum(np.abs(z) - a, 0.0)

lam1 = 0.5                                           # lasso penalty strength λ (tune via CV in practice)
w_lasso = np.zeros(d)                                # initialize coefficients at zero

# Precompute squared norms of columns (useful in coordinate updates)
col_sqnorm = (X ** 2).sum(axis=0)                    # each feature's squared norm (n summed)

# Simple coordinate descent loop (few iterations just for demonstration)
for _ in range(200):                                 # iterate a fixed number of passes
    for j in range(d):                               # update one coordinate (feature) at a time
        # Compute partial residual excluding feature j:
        # r = y - X@w + X[:,j]*w_j  = y - (sum_k X[:,k]*w_k) + X[:,j]*w_j
        r = y - (X @ w_lasso) + X[:, j] * w_lasso[j]
        # Compute "least-squares" target for coordinate j: z_j = (X_j^T r) / (X_j^T X_j)
        z_j = (X[:, j] @ r) / col_sqnorm[j]
        # Soft-threshold to apply L1 penalty; scale by λ/||X_j||^2
        w_lasso[j] = soft_thresh(z_j, lam1 / col_sqnorm[j])

# -----------------------------
# 4) Inspect results
# -----------------------------

print("True weights:  ", np.round(w_true, 3))        # ground-truth for reference
print("Ridge weights: ", np.round(w_ridge, 3))       # ridge shrinks smoothly (rarely exact zeros)
print("Lasso weights: ", np.round(w_lasso, 3))       # lasso often zeros some coefficients exactly
```

### What to expect

* **Ridge** tends to share weight between the **correlated** features $x_1, x_2$ and keep small (non-zero) values, improving stability.
* **Lasso** often **zeros out** one or more coefficients (especially the irrelevant $x_3$). With strong collinearity, it may keep one of $x_1$ or $x_2$ and drop the other, yielding a **sparse** model.

---

## Constrained view (same idea, different lens)

* **Ridge (constrained):** minimize $\|Xw-y\|_2^2$ subject to $\|w\|_2^2 \le t$.
* **Lasso (constrained):** minimize $\|Xw-y\|_2^2$ subject to $\|w\|_1 \le t$.
  
  For every $\lambda$ (penalized form) there exists a corresponding $t$ (constrained form) producing the same solution.

---

## TL;DR

* **Ridge ($\ell_2$)**: smooth shrinkage, stabilizes multicollinearity, better numeric conditioning, **not sparse**.
* **Lasso ($\ell_1$)**: induces **sparsity** (feature selection), great for interpretability, can be unstable under strong collinearity (consider **Elastic Net**).


**Worked examples**

Let $v=(3,-4,1)$.

* $\|v\|_1 = 8$
* $\|v\|_2 = \sqrt{26}\approx 5.099$
* $\|v\|_\infty = 4$

Triangle inequality example with $x=(1,2)$, $y=(-2,1)$:

* $\|x\|_2=\sqrt{5}$, $\|y\|_2=\sqrt{5}$
* $x+y=(-1,3)$, $\|x+y\|_2=\sqrt{10}\le 2\sqrt{5}$ ✓

**NumPy check**

```python
import numpy as np
v = np.array([3,-4,1.0])
l1   = np.sum(np.abs(v))                 # 8.0
l2   = np.linalg.norm(v)                 # 5.099...
linf = np.max(np.abs(v))                 # 4.0
l1, l2, linf
```

# Unit Balls & Norm Shapes

## What is a “unit ball”?

Given a norm $\|\cdot\|$, the **unit ball** is the set of points within distance 1 of the origin:

$$
B = \{ x \in \mathbb{R}^d : \|x\| \le 1 \}.
$$
Different norms define distance differently, so in 2D the unit ball changes shape.

---

## Euclidean norm ($\ell_2$) → **Circle**

**Norm:** $\displaystyle \|x\|_2 = \sqrt{x_1^2 + x_2^2}$

**Unit ball:**

$$
\{(x_1,x_2): \sqrt{x_1^2+x_2^2} \le 1\}
\quad\Longleftrightarrow\quad
x_1^2 + x_2^2 \le 1.
$$

**Shape:** circle centered at the origin (rotationally symmetric).

---

## Manhattan norm ($\ell_1$) → **Diamond**

**Norm:** $\displaystyle \|x\|_1 = |x_1| + |x_2|$

**Unit ball:**

$$
\{(x_1,x_2): |x_1| + |x_2| \le 1\}.
$$

**Shape:** diamond (a square rotated $45^\circ$).  
**Mnemonic:** sum of absolute coordinates = constant → straight-line facets meeting at $(\pm 1,0)$ and $(0,\pm 1)$.

**ML note:** corners on axes encourage **sparsity** when used as a penalty (lasso).

---

## Max norm ($\ell_\infty$) → **Square**

**Norm:** $\displaystyle \|x\|_\infty = \max\{|x_1|, |x_2|\}$

**Unit ball:**

$$
\{(x_1,x_2): \max(|x_1|,|x_2|) \le 1\}
\quad\Longleftrightarrow\quad
|x_1|\le 1 \text{ and } |x_2|\le 1.
$$

**Shape:** axis-aligned square with corners at $(\pm 1, \pm 1)$.
**Mnemonic:** each coordinate is capped by 1 in absolute value → a box.

---

## How to remember

* $\ell_2$ → **circle** (smooth, rotationally symmetric).
* $\ell_1$ → **diamond** (sum of absolutes).
* $\ell_\infty$ → **square** (max/box constraint).

---

## Quick paper checks

* Point $(\tfrac12,\tfrac12)$:

  * $\ell_2$: $\sqrt{\tfrac14+\tfrac14}=\sqrt{\tfrac12}<1$ ✓
  * $\ell_1$: $\tfrac12+\tfrac12=1$ (on boundary) ✓
  * $\ell_\infty$: $\max(\tfrac12,\tfrac12)=\tfrac12<1$ ✓
* Point $(1,0)$: on the boundary of all three.

---

## Why ML cares

* **Constraints/penalties** interact with geometry:

  * $\ell_2$ (ridge): smooth shrinkage, coefficients rarely exactly zero.
  * $\ell_1$ (lasso): corners → solutions often land on axes (exact zeros).
  * $\ell_\infty$: per-feature bound (box constraints), robust max-error criteria.


# Projections

A **projection** maps a vector onto a subspace (line/plane/etc.) so the error (residual) is **orthogonal** to that subspace.

---

## 1) Projection onto a **line** (direction $u$)

**Formula**  
Given $x,u\in\mathbb{R}^d$ with $u\neq 0$:

$$
\mathrm{proj}_u(x)=\frac{x^\top u}{u^\top u}u, \quad
r=x-\mathrm{proj}_u(x), \quad u^\top r = 0.
$$

**Matrix form:** $P_u=\dfrac{uu^\top}{u^\top u}$ so $\mathrm{proj}_u(x)=P_u x$.  
**Properties:** $P_u^\top=P_u$, $P_u^2=P_u$ (symmetric, idempotent), $\mathrm{rank}(P_u)=1$.

### Code + 2D visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# DATA: a point x and a direction u
# ---------------------------

x = np.array([2.5, 3.5], dtype=float)
# x is the vector (a point in 2D) we want to "drop" onto a line.
# Geometrically: imagine a light above x; its shadow on the line (through the origin)
# in direction u is the projection.

u = np.array([2.0, 1.0], dtype=float)  # projection direction
# u defines the line (through the origin) we are projecting onto.
# The "line spanned by u" means all multiples of u: { t * u : t ∈ ℝ }.
# NOTE: u must be nonzero. Its *direction* matters, not its length.

# ---------------------------
# PROJECTION: compute the closest point on the line to x
# ---------------------------

alpha = (x @ u) / (u @ u)
# alpha is a scalar. Intuition:
# - (x @ u) is how much x aligns with u (dot product).
# - (u @ u) is ||u||^2 (squared length of u).
# The ratio answers: "how many 'units of u' best match x along u?"

x_proj = alpha * u
# The actual projection point on the line. It's the multiple of u that lies closest to x.
# This minimizes distance ||x - t*u|| over t. (Set derivative to 0 ⇒ t = alpha.)

r = x - x_proj
# Residual vector from the projection point to x.
# KEY PROPERTY: r is orthogonal (perpendicular) to u (and thus to the line).
# This is what makes x_proj the *closest* point to x on that line.

# ---------------------------
# PLOT: draw the geometry to see the projection
# ---------------------------

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')
# 'equal' aspect so 1 unit on x- and y-axes looks the same (no distortion).

ax.axhline(0, linewidth=1)
ax.axvline(0, linewidth=1)
# Draw axes for reference.

# Line through origin in direction u (visualize the subspace we're projecting onto)
t = np.linspace(-3, 3, 200)
line = np.outer(t, u/np.linalg.norm(u))
# u/||u|| is the unit direction; multiplying by t traces points along the line.

ax.plot(line[:,0], line[:,1], linewidth=1)

# Vectors from the origin (draw x, u, and the projection x_proj)
ax.quiver(0,0, x[0],     x[1],     angles='xy', scale_units='xy', scale=1)
ax.quiver(0,0, u[0],     u[1],     angles='xy', scale_units='xy', scale=1)
ax.quiver(0,0, x_proj[0],x_proj[1],angles='xy', scale_units='xy', scale=1)
# quiver draws arrows. We draw three:
# - x       : the original vector
# - u       : the direction defining the line
# - x_proj  : the projection of x onto the line

# Residual as dashed connector (shows the “perpendicular drop”)
ax.plot([x_proj[0], x[0]], [x_proj[1], x[1]], linestyle='--')
# This dashed segment is r = x - x_proj. It should be perpendicular to u.

ax.set_xlim(-1, 4.5)
ax.set_ylim(-1, 4.5)
ax.set_title("Projection of x onto line spanned by u")
ax.text(x[0], x[1], "  x")
ax.text(u[0], u[1], "  u")
ax.text(x_proj[0], x_proj[1], "  proj_u(x)")

print("alpha:", alpha)
print("u^T r ≈ 0 ?", np.isclose(u @ r, 0.0, atol=1e-10))
# Orthogonality check: dot(u, r) ≈ 0 means r ⟂ u (perpendicular).
# This is THE defining property of an orthogonal projection:
# (x - x_proj) is orthogonal to the subspace (here, the line).

plt.show()
```



---

## 2) Projection onto a **subspace** spanned by columns of $A$

Let $A\in\mathbb{R}^{n\times k}$ have full column rank. The orthogonal projector onto $\mathcal{S}=\mathrm{span}(A)$ is:

$$
P_A = A(A^\top A)^{-1}A^\top,\quad \mathrm{proj}_{\mathcal{S}}(x)=P_A x.
$$

**Properties:** $P_A^\top=P_A$, $P_A^2=P_A$, $\mathrm{rank}(P_A)=k$.  
**Residual orthogonality:** $A^\top(x-P_Ax)=0$.

### Code + 3D plane visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# ---------------------------
# SUBSPACE: a plane in R^3 spanned by two basis vectors u1, u2
# ---------------------------

u1 = np.array([1., 0., 1.])
u2 = np.array([0., 1., 1.])
A  = np.column_stack([u1, u2])  # Shape 3x2: columns are u1 and u2 (the basis of the plane)
# 'span(u1, u2)' is all linear combos a*u1 + b*u2 — geometrically, a plane through the origin.

x  = np.array([2., 1., 0.])     # The vector we will project onto that plane

# ---------------------------
# ORTHOGONAL PROJECTOR onto span(A)
# ---------------------------

P = A @ np.linalg.inv(A.T @ A) @ A.T
# P is the orthogonal projector matrix onto the column space of A (the plane).
# Formula: P_A = A (A^T A)^{-1} A^T  (requires A to have independent columns)
# Why it works: P maps any vector x to the closest point in the subspace span(A).

x_proj = P @ x
# The actual projection of x onto the plane — the point in the plane closest to x (in Euclidean distance).

r = x - x_proj
# Residual (error) vector from the projection point to x.
# KEY PROPERTY: r is orthogonal to the entire subspace (to every column of A).

# ---------------------------
# VISUALIZE the plane and the projection in 3D
# ---------------------------

# Build a grid patch of the plane: a*u1 + b*u2 for a,b in [-2, 2]
s = np.linspace(-2, 2, 20)
t = np.linspace(-2, 2, 20)
S, T = np.meshgrid(s, t)
plane = np.outer(S.ravel(), u1) + np.outer(T.ravel(), u2)  # (400 x 3) points on the plane
Xpl = plane[:,0].reshape(S.shape)
Ypl = plane[:,1].reshape(S.shape)
Zpl = plane[:,2].reshape(S.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xpl, Ypl, Zpl, alpha=0.25, linewidth=0)
# Semi-transparent surface to show the plane spanned by u1 and u2.

# Draw x and its projection from the origin as arrows
ax.quiver(0,0,0, x[0],     x[1],     x[2],     length=1, normalize=False)
ax.quiver(0,0,0, x_proj[0],x_proj[1],x_proj[2],length=1, normalize=False)
# These show the original vector x and the projected vector x_proj.

# Draw the residual as a dashed segment from x_proj to x
ax.plot([x_proj[0], x[0]], [x_proj[1], x[1]], [x_proj[2], x[2]], linestyle='--')
# Geometrically, this is the "perpendicular drop" to the plane.

# Also draw the basis vectors u1 and u2 for reference
ax.quiver(0,0,0, u1[0], u1[1], u1[2], length=1, normalize=False)
ax.quiver(0,0,0, u2[0], u2[1], u2[2], length=1, normalize=False)

ax.set_title("Projection of x onto span(u1, u2)")
ax.text(x[0], x[1], x[2], "x")
ax.text(x_proj[0], x_proj[1], x_proj[2], "proj_A(x)")
ax.set_xlim(-2, 3); ax.set_ylim(-2, 3); ax.set_zlim(-2, 3)

# ---------------------------
# FUNDAMENTAL PROJECTION CHECKS
# ---------------------------

print("Symmetric?", np.allclose(P, P.T))
# Orthogonal projectors are symmetric: P^T = P

print("Idempotent?", np.allclose(P@P, P))
# Projecting twice does nothing new: P^2 = P

print("A^T (x - Px) ≈ 0 ?", np.allclose(A.T @ r, np.zeros(2), atol=1e-10))
# Residual is orthogonal to the subspace: each column of A is orthogonal to r
# (these are the normal equations for projection / least squares)

plt.show()
```

---

## 3) Least squares as a projection

For linear regression $\min_w \|Xw - y\|_2^2$, the fitted values are the projection of $y$ onto the column space of $X$:

$$
\hat{y} = P_X y, \quad P_X = X(X^\top X)^{-1}X^\top, \quad X^\top(y-\hat{y})=0.
$$

### Code + 2D picture (data, regression line, vertical residuals)

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n = 20
x1 = np.linspace(0, 1, n)
y  = 1.0 + 2.0*x1 + 0.2*rng.normal(size=n)  # noisy line

# Design matrix with intercept
X = np.column_stack([np.ones(n), x1])  # n x 2
# Closed-form OLS
w = np.linalg.solve(X.T @ X, X.T @ y)
y_hat = X @ w

# Projector P_X (for illustration only)
PX = X @ np.linalg.inv(X.T @ X) @ X.T
res = y - y_hat
orth_check = np.allclose(X.T @ res, np.zeros(2), atol=1e-10)

# Plot
fig, ax = plt.subplots()
ax.scatter(x1, y, label="data")
ax.plot(x1, y_hat, label="fit", linewidth=2)

# residuals as vertical lines
for xi, yi, yhi in zip(x1, y, y_hat):
    ax.plot([xi, xi], [yi, yhi], linestyle='--', linewidth=1)

ax.set_title("Least squares = projection of y onto col(X)")
ax.legend()
print("Normal equations satisfied: X^T (y - ŷ) ≈ 0 ?", orth_check)
plt.show()
```

---

## 4) Pythagorean decomposition with a projector

For an orthogonal projector $P$ and any $x$:

$$
x = Px + (I-P)x,\quad (Px)^\top((I-P)x)=0,\quad
\|x\|_2^2=\|Px\|_2^2+\|(I-P)x\|_2^2.
$$

### Code + 2D visualization (decompose $x$ into parallel + perpendicular parts)

```python
import numpy as np
import matplotlib.pyplot as plt

# Choose a line direction u and a point x in R^2
u = np.array([1.0, 2.0])
x = np.array([3.0, 1.0])
P = np.outer(u, u) / (u @ u)

x_par = P @ x             # parallel component
x_perp = x - x_par        # perpendicular component

# Checks
print("Orthogonality:", np.isclose(x_par @ x_perp, 0.0, atol=1e-10))
lhs = np.linalg.norm(x)**2
rhs = np.linalg.norm(x_par)**2 + np.linalg.norm(x_perp)**2
print("Pythagoras:", np.isclose(lhs, rhs, atol=1e-10))

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')
ax.axhline(0, linewidth=1); ax.axvline(0, linewidth=1)

# line for subspace
t = np.linspace(-3, 3, 200)
line = np.outer(t, u/np.linalg.norm(u))
ax.plot(line[:,0], line[:,1], linewidth=1)

# vectors
ax.quiver(0,0, x[0],            x[1],            angles='xy', scale_units='xy', scale=1, label='x')
ax.quiver(0,0, x_par[0],        x_par[1],        angles='xy', scale_units='xy', scale=1, label='Px')
ax.quiver(x_par[0], x_par[1], x_perp[0], x_perp[1], angles='xy', scale_units='xy', scale=1, label='(I-P)x')

ax.set_title("Pythagorean split: x = Px + (I-P)x")
ax.legend()
plt.show()
```

---

## 5) Stable computation tips (no plot)

* Avoid explicit inverses in practice; prefer **QR**/**SVD**.

  * If $A=QR$ (reduced QR), then $P_A = QQ^\top$.
  * If $A=U\Sigma V^\top$ (SVD), projector onto $\mathrm{col}(A)$ is $U_r U_r^\top$ (keep columns with nonzero singular values).

```python
# Given A (n x k)
Q, R = np.linalg.qr(A, mode='reduced')
P_qr = Q @ Q.T

U, S, Vt = np.linalg.svd(A, full_matrices=False)
r = np.sum(S > 1e-12)       # numerical rank
U_r = U[:, :r]
P_svd = U_r @ U_r.T

np.allclose(P_qr, P_svd)
```

---

### Recap

* **Line projection:** $P_u=\dfrac{uu^\top}{u^\top u}$.
* **Subspace projection:** $P_A=A(A^\top A)^{-1}A^\top$.
* **Geometric heart of ML:** least squares, PCA, and many denoising/compression methods are (or use) projections.


---

## 2) Matrices & Linear Maps

**What:** A matrix $A$ represents a linear map $x\mapsto Ax$. Shapes: $(n\times d)(d\times k)=(n\times k)$.  
**Why ML:** data matrix $X\in\mathbb{R}^{n\times d}$ (rows=samples, cols=features); predictions $Xw$.  
**Remember:** Column space (span of columns); rank = dimension of column space.  
**Pitfall:** Row vs column orientation changes shapes.

---

## 3) Orthogonality & Projections

**What:** $x\perp y\iff x^\top y=0$. Projection onto direction $u$: $P=\dfrac{uu^\top}{u^\top u}$.  
**Why ML:** least squares is an orthogonal projection of $y$ onto span of $X$'s columns.  
**Remember:** Projection matrices are symmetric & idempotent: $P^\top=P$, $P^2=P$.

```python
# NumPy check
u = rng.normal(size=4)
P = np.outer(u,u)/(u@u)
np.allclose(P,P.T), np.allclose(P@P,P)
```

---

## 4) Least Squares (Linear Regression)

**Problem:** $\min_w \|Xw - y\|_2^2$.  
**Solution:** normal equations $X^\top X w = X^\top y$; if full column rank, $w=(X^\top X)^{-1}X^\top y$.  
**Why ML:** baseline predictor; building block for GLMs.  
**Numerics:** prefer `np.linalg.lstsq(X,y)` or QR/SVD over explicit inversion.

---

## 5) Eigenvalues/Eigenvectors (Symmetric Case)

**What:** $A v=\lambda v$. For symmetric $A$, real eigenvalues, orthonormal eigenvectors.  
**Why ML:** covariance matrices, Laplacians, power iteration, spectral clustering.  
**Remember:** For PSD $A\succeq 0$, all $\lambda_i\ge 0$.

```python
# NumPy check
A = rng.normal(size=(4,4)); A = (A+A.T)/2
w,V = np.linalg.eigh(A)  # eigenvalues (w) & eigenvectors (V)
```

---