"""Synthetic 2D furnace benchmark (vertical mid-plane), a stand-in for CFD data.

Steady energy balance (units: K, metres scaled so width = 1, height = 4):
    uy(x) dT/dy = alpha * (d2T/dx2 + d2T/dy2) + q(x, y; tilt) - beta * (T - T_wall)
BCs: side walls T = T_wall (water walls), bottom T = T_in, top dT/dy = 0 (outlet).
q = two Gaussian heat-release zones whose height moves with burner tilt,
mimicking the trend reported for Fig. 6 of Xue et al. (2024).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

W, H = 1.0, 4.0
NX, NY = 61, 481
ALPHA, BETA = 0.01, 0.30          # effective diffusivity, linearised radiative loss
T_WALL, T_IN = 650.0, 700.0
TILTS = [-30, -20, -10, 0, 10, 20, 30]
FLAME_BAND = (0.6, 3.1)           # heights where combustion heat release may be non-zero

def uy(x, tilt=0.0):
    # upward velocity, faster in the core; tilt skews the profile (stands in for cold-flow CFD)
    t = tilt / 30.0
    return 0.5 + 2.0 * x * (1 - x) + 0.35 * t * (x - 0.5)

def q(x, y, tilt):
    t = tilt / 30.0
    y1, y2 = 1.55 + 0.55 * t, 2.10 + 0.35 * t          # flame zones move with tilt
    xc = 0.5 + 0.10 * np.sin(np.pi * t)                 # fireball sways sideways (non-monotonic)
    a1, a2 = 3400.0 * (1 - 0.2 * t), 3000.0 * (0.6 + 0.35 * t)
    gx = np.exp(-((x - xc) ** 2) / (2 * 0.18 ** 2))
    return gx * (a1 * np.exp(-((y - y1) ** 2) / (2 * 0.11 ** 2)) + a2 * np.exp(-((y - y2) ** 2) / (2 * 0.11 ** 2)))

def solve(tilt, alpha=ALPHA, beta=BETA):
    x = np.linspace(0, W, NX); y = np.linspace(0, H, NY)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    n = NX * NY; idx = lambda i, j: i * NY + j
    rows, cols, vals = [], [], []; b = np.zeros(n)
    def add(r, c, v): rows.append(r); cols.append(c); vals.append(v)
    for i in range(NX):
        for j in range(NY):
            k = idx(i, j)
            if i == 0 or i == NX - 1:
                add(k, k, 1.0); b[k] = T_WALL; continue
            if j == 0:
                add(k, k, 1.0); b[k] = T_IN; continue
            if j == NY - 1:                      # zero-gradient outlet
                add(k, k, 1.0); add(k, idx(i, j - 1), -1.0); continue
            u = uy(x[i], tilt)
            # central convection + central diffusion (2nd order; cell Peclet < 2 on this grid)
            add(k, k, 2 * alpha / dx**2 + 2 * alpha / dy**2 + beta)
            add(k, idx(i, j - 1), -u / (2 * dy) - alpha / dy**2)
            add(k, idx(i, j + 1), u / (2 * dy) - alpha / dy**2)
            add(k, idx(i - 1, j), -alpha / dx**2)
            add(k, idx(i + 1, j), -alpha / dx**2)
            b[k] = q(x[i], y[j], tilt) + beta * T_WALL
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    T = spla.spsolve(A, b).reshape(NX, NY)
    return X, Y, T

def make_dataset():
    out = {}
    for t in TILTS:
        X, Y, T = solve(t)
        out[t] = T
    return X, Y, out

if __name__ == "__main__":
    X, Y, D = make_dataset()
    for t in TILTS:
        T = D[t]; j = np.unravel_index(T.argmax(), T.shape)
        print(f"tilt {t:+3d}: min {T.min():.0f} K  max {T.max():.0f} K at y={Y[j]:.2f}  outlet mean {T[:, -1].mean():.0f} K")
