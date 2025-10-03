# Expanded visuals + runnable examples for Vectors, Norms, and Inner Products
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ---------- 1) Vector geometry: angle & dot product ----------
x = np.array([3.0, 1.5])
y = np.array([1.0, 3.0])

def plot_vectors_and_angle(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_aspect('equal', adjustable='box')
    ax.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1)
    ax.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1)
    ax.set_xlim(-0.5, max(4, x[0], y[0]) + 0.5)
    ax.set_ylim(-0.5, max(4, x[1], y[1]) + 0.5)
    ax.set_title("Vectors x and y with angle θ (geometry of the dot product)")

    # draw small arc for the angle between x and y at the origin
    # compute unit vectors
    x_u = x / np.linalg.norm(x)
    y_u = y / np.linalg.norm(y)
    # angle between them
    cos_th = np.clip(x_u @ y_u, -1, 1)
    theta = np.arccos(cos_th)
    # parametric arc from x_u to y_u (shorter arc)
    npts = 100
    # choose rotation direction by sign of cross product z-component
    cross_z = x_u[0]*y_u[1] - x_u[1]*y_u[0]
    if cross_z >= 0:
        angles = np.linspace(np.arctan2(x_u[1], x_u[0]), np.arctan2(y_u[1], y_u[0]), npts)
    else:
        angles = np.linspace(np.arctan2(x_u[1], x_u[0]), np.arctan2(y_u[1], y_u[0]) + 2*np.pi, npts)
        # wrap to shorter arc if needed
        if (angles[-1] - angles[0]) > np.pi:
            angles = np.linspace(np.arctan2(y_u[1], y_u[0]), np.arctan2(x_u[1], x_u[0]), npts)
    r = 0.8
    arc_x = r * np.cos(angles)
    arc_y = r * np.sin(angles)
    ax.plot(arc_x, arc_y, linewidth=2)
    ax.text(arc_x[len(arc_x)//2], arc_y[len(arc_y)//2], "θ")

    # Annotate norms and dot product
    ax.text(x[0], x[1], "  x", fontsize=10)
    ax.text(y[0], y[1], "  y", fontsize=10)

    plt.show()

plot_vectors_and_angle(x, y)

# Numeric checks for the angle and dot product
dot_xy = float(x @ y)
cos_theta = dot_xy / (np.linalg.norm(x) * np.linalg.norm(y))
theta_deg = float(np.degrees(np.arccos(np.clip(cos_theta, -1, 1))))
print("Dot product x·y =", round(dot_xy, 3))
print("||x||_2 =", round(np.linalg.norm(x), 3), "||y||_2 =", round(np.linalg.norm(y), 3))
print("cos θ =", round(cos_theta, 4), "θ (deg) =", round(theta_deg, 2))


# ---------- 2) Projection of x onto a direction u ----------
u = np.array([2.0, 1.0])  # projection direction
z = np.array([2.0, 3.0])  # vector to project

def plot_projection(z, u):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_aspect('equal', adjustable='box')

    # line through origin in direction u
    t = np.linspace(-3, 3, 100)
    line = np.outer(t, u / np.linalg.norm(u))
    ax.plot(line[:,0], line[:,1], linewidth=1)

    # projection
    alpha = (z @ u) / (u @ u)
    z_proj = alpha * u

    # draw vectors
    ax.quiver(0, 0, z[0], z[1], angles='xy', scale_units='xy', scale=1)
    ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1)
    ax.quiver(0, 0, z_proj[0], z_proj[1], angles='xy', scale_units='xy', scale=1)

    # dashed segment from projection to z (residual)
    ax.plot([z_proj[0], z[0]], [z_proj[1], z[1]], linestyle='--', linewidth=1)

    ax.set_xlim(-1, 4.5)
    ax.set_ylim(-1, 4.5)
    ax.set_title("Projection of z onto direction u")

    ax.text(z[0], z[1], "  z")
    ax.text(u[0], u[1], "  u")
    ax.text(z_proj[0], z_proj[1], "  proj_u(z)")

    plt.show()

plot_projection(z, u)

alpha = (z @ u) / (u @ u)
z_proj = alpha * u
residual = z - z_proj
orthogonality = float(z_proj @ residual)
print("alpha =", round(alpha, 4))
print("proj_u(z) =", np.round(z_proj, 3))
print("residual z - proj =", np.round(residual, 3))
print("Orthogonality check proj·residual ≈ 0:", np.isclose(orthogonality, 0.0, atol=1e-10), "value =", orthogonality)


# ---------- 3) Norm balls in R^2: l1, l2, l_infty ----------
def plot_norm_balls():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    # l2 unit circle
    t = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), linewidth=2, label="l2 unit ball")

    # l_infty unit ball (square)
    square = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])
    ax.plot(square[:,0], square[:,1], linewidth=2, label="linf unit ball")

    # l1 unit ball (diamond)
    # Parametric diamond: |x| + |y| = 1
    # We'll draw as straight lines between corners
    diamond = np.array([[0,1],[1,0],[0,-1],[-1,0],[0,1]])
    ax.plot(diamond[:,0], diamond[:,1], linewidth=2, label="l1 unit ball")

    ax.legend()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Unit balls: l1 (diamond), l2 (circle), linf (square)")
    plt.show()

plot_norm_balls()

# ---------- 4) Tiny experiments to run quickly ----------
# Angle sensitivity: scale one vector, cosine stays the same
x1 = np.array([2.0, 1.0, -1.0])
y1 = np.array([1.0, -1.0, 2.0])
cos1 = (x1 @ y1) / (np.linalg.norm(x1)*np.linalg.norm(y1))
cos2 = (3*x1 @ (0.5*y1)) / (np.linalg.norm(3*x1)*np.linalg.norm(0.5*y1))
print("Cosine invariance to positive scaling:", float(cos1), float(cos2))

# Norm comparisons on the same vector
v = np.array([3.0, -4.0, 1.0])
l1 = np.sum(np.abs(v))
l2 = np.linalg.norm(v)
linf = np.max(np.abs(v))
print("Norms on v=[3,-4,1]: ||.||1, ||.||2, ||.||inf =", l1, l2, linf)

# Cauchy–Schwarz numeric check
x2 = rng.normal(size=5); y2 = rng.normal(size=5)
lhs = abs(x2 @ y2)
rhs = np.linalg.norm(x2)*np.linalg.norm(y2)
print("Cauchy–Schwarz holds (lhs ≤ rhs):", lhs <= rhs, "lhs=", round(lhs,4), "rhs=", round(rhs,4))
