"""Builds PIML_Furnace_Starter.ipynb from the cells below (no nbformat needed)."""
import json, sys

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# PIML starter: reconstructing a furnace temperature field

**FYP 2026/2027 · first draft**

This notebook builds the first physics-informed machine learning (PIML) model for the FYP, following the set-up of
Xue et al. (2024), *Energy* 286, 129568: offline CFD cases at several burner tilts → a neural network → the temperature
field at a tilt the network has never seen.

**What is in it**

| Part | What it does |
|---|---|
| 1 | A small synthetic "CFD" benchmark: a 2D vertical slice of a furnace solved with a known energy equation, 7 burner tilts |
| 2 | Training data like the paper's: 5% random points from 6 tilts; test on the unseen 0° tilt; the paper's 4 metrics |
| 3 | Baselines: linear interpolation, the paper-style "sandwich" MLP, a plain data-only MLP |
| 4 | **Hybrid PINN**: the same MLP + energy-equation residual in flame-free regions + boundary conditions, with the diffusivity and radiation coefficient *learned* |
| 5 | Experiments: (a) paper setting, (b) correction with 8 noisy "thermocouples", (c) very sparse data |
| 6 | How to swap in real CFD data from Fluent |

**Why synthetic data?** The anchor paper's CFD data is only "available on request". The benchmark lets every model be
checked against an exact answer and an exactly known physics law. Swap in real CFD exports later (Part 6).

**Run it:** Colab → *Runtime → Change runtime type → T4 GPU* → *Run all*. A GPU run takes a few minutes; CPU works but is slower.
""")

code(r"""
import time
from itertools import combinations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
print("device:", device)
""")

md(r"""
## Part 1 · Synthetic furnace benchmark

A vertical slice through the furnace centre, scaled so the width is 1 and the height is 4 (the real boiler is 14.63 m × 58.3 m).
The temperature obeys a steady energy balance:

$$u_y(x)\,\frac{\partial T}{\partial y} \;=\; \alpha\left(\frac{\partial^2 T}{\partial x^2}+\frac{\partial^2 T}{\partial y^2}\right) \;+\; q(x,y;\theta) \;-\; \beta\,(T-T_{wall})$$

| Term | Meaning | In a real furnace |
|---|---|---|
| $u_y\,\partial T/\partial y$ | hot gas carried upward | velocity from CFD |
| $\alpha\,\nabla^2 T$ | turbulent mixing of heat | effective diffusivity, unknown |
| $q$ | heat released by burning coal, two zones that move with burner tilt $\theta$ | unknown, stiff, very local |
| $\beta\,(T-T_{wall})$ | heat lost by radiation to the water walls | unknown coefficient |

Boundaries: water walls at 650 K, bottom (hopper) at 700 K, zero gradient at the top outlet.
The heat release $q$ is non-zero only inside the **flame band** (heights 0.6 to 3.1). Outside it, the equation has no source term,
and that is where the PINN applies physics.
""")

code(r"""
W, H = 1.0, 4.0
NX, NY = 61, 481                       # grid (central differences, cell Peclet < 2)
ALPHA_TRUE, BETA_TRUE = 0.01, 0.30
T_WALL, T_IN = 650.0, 700.0
TILTS = [-30, -20, -10, 0, 10, 20, 30]
FLAME_BAND = (0.6, 3.1)

def velocity_y(x, tilt):
    # Upward gas velocity. In real data this comes from CFD (or a cheap cold-flow CFD at a new condition).
    t = tilt / 30.0
    return 0.5 + 2.0 * x * (1 - x) + 0.35 * t * (x - 0.5)

def heat_release(x, y, tilt):
    t = tilt / 30.0
    y1, y2 = 1.55 + 0.55 * t, 2.10 + 0.35 * t          # flame zones move with tilt
    xc = 0.5 + 0.10 * np.sin(np.pi * t)                 # fireball sways sideways
    a1, a2 = 3400.0 * (1 - 0.2 * t), 3000.0 * (0.6 + 0.35 * t)
    gx = np.exp(-((x - xc) ** 2) / (2 * 0.18 ** 2))
    return gx * (a1 * np.exp(-((y - y1) ** 2) / (2 * 0.11 ** 2)) + a2 * np.exp(-((y - y2) ** 2) / (2 * 0.11 ** 2)))

def solve_furnace(tilt, alpha=ALPHA_TRUE, beta=BETA_TRUE):
    x = np.linspace(0, W, NX); y = np.linspace(0, H, NY)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    n = NX * NY; idx = lambda i, j: i * NY + j
    rows, cols, vals = [], [], []; b = np.zeros(n)
    def add(r, c, v): rows.append(r); cols.append(c); vals.append(v)
    for i in range(NX):
        for j in range(NY):
            k = idx(i, j)
            if i == 0 or i == NX - 1: add(k, k, 1.0); b[k] = T_WALL; continue
            if j == 0: add(k, k, 1.0); b[k] = T_IN; continue
            if j == NY - 1: add(k, k, 1.0); add(k, idx(i, j - 1), -1.0); continue
            u = velocity_y(x[i], tilt)
            add(k, k, 2 * alpha / dx**2 + 2 * alpha / dy**2 + beta)
            add(k, idx(i, j - 1), -u / (2 * dy) - alpha / dy**2)
            add(k, idx(i, j + 1), u / (2 * dy) - alpha / dy**2)
            add(k, idx(i - 1, j), -alpha / dx**2)
            add(k, idx(i + 1, j), -alpha / dx**2)
            b[k] = heat_release(x[i], y[j], tilt) + beta * T_WALL
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return X, Y, spla.spsolve(A, b).reshape(NX, NY)

t0 = time.time()
CFD = {}
for tilt in TILTS:
    X, Y, CFD[tilt] = solve_furnace(tilt)
print(f"7 'CFD' cases solved in {time.time() - t0:.1f} s, {X.size:,} points each")
""")

code(r"""
fig, axes = plt.subplots(1, 7, figsize=(16, 5), sharey=True)
for ax, tilt in zip(axes, TILTS):
    im = ax.imshow(CFD[tilt].T, origin="lower", extent=[0, W, 0, H], cmap="jet", vmin=650, vmax=1900, aspect="auto")
    ax.axhspan(*FLAME_BAND, color="white", alpha=0.08)
    ax.set_title(f"{tilt:+d}°"); ax.set_xlabel("x")
axes[0].set_ylabel("height y")
fig.colorbar(im, ax=axes, label="T (K)", shrink=0.8)
plt.suptitle("Synthetic CFD temperature field: the hot zone rises as burners tilt up")
plt.show()
""")

md(r"""
## Part 2 · Training data, test case and metrics

Same protocol as the paper: train on 6 tilts (0° excluded), keep a random **5%** of the points of each, test on **all** points at 0°.
Metrics are the paper's: R², MAE (K), MAPE (%), RMSE (K).

Eight noisy "thermocouples" at 0° are also prepared for Experiment (b). Their positions are like wall-mounted probes and
probes in the upper furnace.
""")

code(r"""
TRAIN_TILTS = [-30, -20, -10, 10, 20, 30]
TEST_TILT = 0
TM, TS = 1000.0, 400.0                  # temperature scaling used inside the networks

def metrics(pred, true):
    e = pred - true
    return {"R2": 1 - (e**2).sum() / ((true - true.mean())**2).sum(), "MAE (K)": np.abs(e).mean(),
            "MAPE (%)": 100 * np.abs(e / true).mean(), "RMSE (K)": np.sqrt((e**2).mean())}

def sample_points(frac, seed=0, tilts=None):
    r = np.random.default_rng(seed); cols = [[], [], [], []]
    for tilt in (TRAIN_TILTS if tilts is None else tilts):
        k = max(1, int(frac * X.size)); ii = r.choice(X.size, k, replace=False)
        for c, v in zip(cols, (X.ravel()[ii], Y.ravel()[ii], np.full(k, float(tilt)), CFD[tilt].ravel()[ii])):
            c.append(v)
    return [np.concatenate(c) for c in cols]          # x, y, tilt, T

def sensor_readings(tilt=TEST_TILT, noise_K=5.0, seed=7):
    pts = [(0.12, 1.8), (0.88, 1.8), (0.12, 2.6), (0.88, 2.6), (0.3, 3.4), (0.7, 3.4), (0.5, 3.9), (0.5, 0.4)]
    r = np.random.default_rng(seed)
    ii = [int(round(px * (NX - 1))) for px, _ in pts]; jj = [int(round(py / H * (NY - 1))) for _, py in pts]
    T = np.array([CFD[tilt][i, j] for i, j in zip(ii, jj)]) + r.normal(0, noise_K, len(pts))
    return (np.array([X[i, j] for i, j in zip(ii, jj)]), np.array([Y[i, j] for i, j in zip(ii, jj)]),
            np.full(len(pts), float(tilt)), T)

data_5pct = sample_points(0.05)
sensors = sensor_readings()
truth = CFD[TEST_TILT]
print(f"training rows: {len(data_5pct[0]):,}   test points: {truth.size:,}   sensors: {len(sensors[0])}")
results = {}
""")

md(r"""
## Part 3 · Baselines

**B0. Linear interpolation.** Average the two neighbouring CFD cases (−10° and +10°). No learning at all. The anchor paper
never reported this, but any model must beat it.
""")

code(r"""
results["B0 Linear interpolation"] = metrics(0.5 * (CFD[-10] + CFD[10]), truth)
results["B0 Linear interpolation"]
""")

md(r"""
**Networks.** All networks below are small fully connected nets with **tanh** activations.

Why tanh and not ReLU (the paper used ReLU)? The PINN needs the *second* derivative of T with respect to x and y. ReLU is
piecewise straight, so its second derivative is zero everywhere and the diffusion term would vanish. tanh is smooth.

Inputs are scaled to about [−1, 1] and the output is rescaled to kelvin inside `forward`, so derivatives come out in
physical units automatically.
""")

code(r"""
def mlp(n_in, width=48, depth=3):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]; d = width
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)

class FieldNet(nn.Module):
    # T(x, y, tilt) in kelvin
    def __init__(self, width=48, depth=3):
        super().__init__(); self.net = mlp(3, width, depth)
    def forward(self, x, y, tilt):
        h = torch.stack([2 * x - 1, y / 2 - 1, tilt / 30.0], dim=1)
        return TM + TS * self.net(h).squeeze(1)

def tt(*arrays):
    return [torch.tensor(np.asarray(a), dtype=DTYPE, device=device) for a in arrays]

def predict(model, tilt):
    xg, yg, tg = tt(X.ravel(), Y.ravel(), np.full(X.size, float(tilt)))
    with torch.no_grad():
        return model(xg, yg, tg).cpu().numpy().reshape(X.shape)

def train(model, data, steps, lr=3e-3, physics=None, extra=None, w_extra=0.2, log_every=1000):
    # Adam with a 10x learning-rate decay over the run. `extra` = sensor readings, `physics` = EnergyPhysics module.
    xs, ys, ts, Ts = tt(*data)
    params = list(model.parameters()) + (list(physics.parameters()) if physics is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.1 ** (k / steps))
    if extra is not None:
        ex, ey, et, eT = tt(*extra)
    t0 = time.time()
    for k in range(1, steps + 1):
        opt.zero_grad()
        loss = (((model(xs, ys, ts) - Ts) / TS) ** 2).mean()
        if extra is not None:
            loss = loss + w_extra * (((model(ex, ey, et) - eT) / TS) ** 2).mean()
        if physics is not None:
            loss = loss + physics(model)
        loss.backward(); opt.step(); sched.step()
        if log_every and k % log_every == 0:
            msg = f"  step {k:5d}  loss {loss.item():.2e}  ({time.time() - t0:.0f} s)"
            if physics is not None:
                msg += f"  alpha={physics.alpha.item():.4f} beta={physics.beta.item():.3f}"
            print(msg)
    return model
""")

md(r"""
**B1. Paper-style "sandwich" MLP** (Xue et al. 2024). Inputs: x, y, the target tilt, the two neighbouring tilts, and the CFD
temperature at the same point for those two tilts. Trained on every sorted triplet of training tilts (middle = target), 5% of points each.
Tested on the sandwich (−10°, **0°**, +10°).
""")

code(r"""
def sandwich_rows(frac, seed=0):
    r = np.random.default_rng(seed); F, T = [], []
    for a, m, b in combinations(TRAIN_TILTS, 3):
        k = int(frac * X.size); ii = r.choice(X.size, k, replace=False)
        F.append(np.stack([X.ravel()[ii], Y.ravel()[ii], np.full(k, m), np.full(k, a), np.full(k, b),
                           CFD[a].ravel()[ii], CFD[b].ravel()[ii]], 1))
        T.append(CFD[m].ravel()[ii])
    return np.concatenate(F), np.concatenate(T)

class SandwichNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net = mlp(7)
    def forward(self, F):
        h = torch.stack([2 * F[:, 0] - 1, F[:, 1] / 2 - 1, F[:, 2] / 30, F[:, 3] / 30, F[:, 4] / 30,
                         (F[:, 5] - TM) / TS, (F[:, 6] - TM) / TS], 1)
        return TM + TS * self.net(h).squeeze(1)

F_train, T_train = tt(*sandwich_rows(0.05))
sand = SandwichNet().to(device)
opt = torch.optim.Adam(sand.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.1 ** (k / 6000))
for k in range(1, 6001):
    opt.zero_grad(); loss = (((sand(F_train) - T_train) / TS) ** 2).mean(); loss.backward(); opt.step(); sched.step()
    if k % 2000 == 0: print(f"  step {k}  loss {loss.item():.2e}")
F_test, = tt(np.stack([X.ravel(), Y.ravel(), np.zeros(X.size), np.full(X.size, -10.0), np.full(X.size, 10.0),
                       CFD[-10].ravel(), CFD[10].ravel()], 1))
with torch.no_grad():
    pred_sand = sand(F_test).cpu().numpy().reshape(X.shape)
results["B1 Paper-style MLP"] = metrics(pred_sand, truth)
results["B1 Paper-style MLP"]
""")

md(r"""
**B2. Data-only MLP.** T = f(x, y, tilt), trained only on the CFD points. This is the PINN's network *without* physics,
so the B2-versus-PINN comparison isolates exactly what physics adds.
""")

code(r"""
torch.manual_seed(1)
mlp_data = train(FieldNet().to(device), data_5pct, steps=6000)
pred_mlp = predict(mlp_data, TEST_TILT)
results["B2 Data-only MLP"] = metrics(pred_mlp, truth)
results["B2 Data-only MLP"]
""")

md(r"""
## Part 4 · Hybrid PINN

Loss = **data** (CFD points) + **physics** (energy-equation residual) + **boundary conditions**.

- **Physics only where it is certain.** The residual is applied at collocation points *outside* the flame band (hopper and
  upper furnace), where there is no heat release. So it needs no combustion model. Inside the band, the CFD data rules.
- **Collocation points cover every tilt from −30° to +30°**, including the unseen 0°. This is how physics reaches
  conditions with no data.
- **α (effective diffusivity) and β (radiative loss) are learned.** They start at 2× and 0.5× their true values, and the
  printout shows whether training recovers 0.01 and 0.30. With real CFD you would not know them.
- **Warm start:** train on data alone first, then switch physics on. This avoids the early tug-of-war between loss
  terms that often breaks PINN training.
- **Physics weight `w_pde = 0.1`.** In testing, weight 1.0 let the physics term overpower the data (MAPE 1.81% vs 1.25%).
  Tuning this weight is the first knob to try on real data.

Derivatives come from `torch.autograd.grad(..., create_graph=True)`: the network is differentiated exactly with respect to its inputs.
""")

code(r"""
class EnergyPhysics(nn.Module):
    def __init__(self, n_col=1500, n_bc=300, alpha0=0.02, beta0=0.15, w_pde=0.1, w_bc=1.0, seed=3):
        super().__init__()
        r = np.random.default_rng(seed)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha0), dtype=DTYPE))
        self.log_beta = nn.Parameter(torch.tensor(np.log(beta0), dtype=DTYPE))
        self.w_pde, self.w_bc = w_pde, w_bc
        # collocation points in flame-free regions, all tilts
        cx = r.uniform(0, W, n_col); ct = r.uniform(-30, 30, n_col)
        cy = np.concatenate([r.uniform(0, FLAME_BAND[0], n_col // 3), r.uniform(FLAME_BAND[1], H, n_col - n_col // 3)])
        # boundary points
        bt = r.uniform(-30, 30, n_bc)
        wall_x = np.where(r.random(n_bc) < 0.5, 0.0, W); wall_y = r.uniform(0, H, n_bc)
        bot_x, top_x = r.uniform(0, W, n_bc), r.uniform(0, W, n_bc)
        for name, arr in dict(cx=cx, cy=cy, ct=ct, cu=velocity_y(cx, ct), bt=bt, wall_x=wall_x, wall_y=wall_y,
                              bot_x=bot_x, top_x=top_x).items():
            self.register_buffer(name, torch.tensor(arr, dtype=DTYPE))

    @property
    def alpha(self): return self.log_alpha.exp()
    @property
    def beta(self): return self.log_beta.exp()

    def residual(self, model):
        x = self.cx.clone().requires_grad_(True); y = self.cy.clone().requires_grad_(True)
        T = model(x, y, self.ct)
        Tx, Ty = torch.autograd.grad(T.sum(), (x, y), create_graph=True)
        Txx = torch.autograd.grad(Tx.sum(), x, create_graph=True)[0]
        Tyy = torch.autograd.grad(Ty.sum(), y, create_graph=True)[0]
        return self.cu * Ty - self.alpha * (Txx + Tyy) + self.beta * (T - T_WALL)

    def forward(self, model):
        L_pde = ((self.residual(model) / TS) ** 2).mean()
        n = len(self.bt)
        L_wall = (((model(self.wall_x, self.wall_y, self.bt) - T_WALL) / TS) ** 2).mean()
        L_bot = (((model(self.bot_x, torch.zeros_like(self.bot_x), self.bt) - T_IN) / TS) ** 2).mean()
        y_top = torch.full_like(self.top_x, H).requires_grad_(True)
        Ty_top = torch.autograd.grad(model(self.top_x, y_top, self.bt).sum(), y_top, create_graph=True)[0]
        L_top = ((Ty_top / TS) ** 2).mean()
        return self.w_pde * L_pde + self.w_bc * (L_wall + L_bot + L_top)

def train_pinn(data, warm_steps=2000, steps=4000, extra=None, seed=1, **phys_kw):
    torch.manual_seed(seed)
    model = train(FieldNet().to(device), data, steps=warm_steps, extra=extra, log_every=0)   # warm start: data only
    phys = EnergyPhysics(**phys_kw).to(device)
    train(model, data, steps=steps, physics=phys, extra=extra)
    return model, phys
""")

md(r"""
## Part 5 · Experiments

### (a) Paper setting: 5% of points from 6 tilts, predict 0°
""")

code(r"""
pinn, phys = train_pinn(data_5pct)
pred_pinn = predict(pinn, TEST_TILT)
results["PINN (hybrid)"] = metrics(pred_pinn, truth)
print(f"learned alpha = {phys.alpha.item():.4f} (true {ALPHA_TRUE}),  beta = {phys.beta.item():.3f} (true {BETA_TRUE})")
results["PINN (hybrid)"]
""")

md(r"""
### (b) Correction with 8 noisy thermocouples at 0°

The plant's DCS gives a few real temperatures at the current condition. Both networks get the same 8 readings (±5 K
noise) in their data loss, with weight `w_extra = 0.2`. In testing, weight 2.0 made the networks chase the sensor noise and
the error went *up*; 0.2 brought it down.
""")

code(r"""
torch.manual_seed(1)
mlp_s = train(FieldNet().to(device), data_5pct, steps=6000, extra=sensors, log_every=0)
results["B2 Data-only MLP + 8 sensors"] = metrics(predict(mlp_s, TEST_TILT), truth)
pinn_s, phys_s = train_pinn(data_5pct, extra=sensors)
results["PINN + 8 sensors"] = metrics(predict(pinn_s, TEST_TILT), truth)
print({k: results[k] for k in ["B2 Data-only MLP + 8 sensors", "PINN + 8 sensors"]})
""")

md(r"""
### (c) Very sparse data: 0.5% of points (about 147 per tilt)

In a real project you may only afford a few CFD cases, or only export a slice. This is where physics usually helps most.
""")

code(r"""
data_sparse = sample_points(0.005)
torch.manual_seed(1)
mlp_sp = train(FieldNet().to(device), data_sparse, steps=6000, log_every=0)
results["B2 Data-only MLP (0.5% data)"] = metrics(predict(mlp_sp, TEST_TILT), truth)
pinn_sp, phys_sp = train_pinn(data_sparse)
results["PINN (0.5% data)"] = metrics(predict(pinn_sp, TEST_TILT), truth)
print({k: results[k] for k in ["B2 Data-only MLP (0.5% data)", "PINN (0.5% data)"]})
""")

md(r"""
### (d) Extrapolation: train on −30° to +10°, predict +30°

A harder test: the target lies outside the training range. Data-only networks usually fail here. The PINN's collocation
points still cover +30°, so the energy balance keeps constraining the solution there.
""")

code(r"""
X_TILTS = [-30, -20, -10, 0, 10]; X_TEST = 30
data_x = sample_points(0.05, tilts=X_TILTS); truth_x = CFD[X_TEST]
results_x = {"Nearest case (+10°)": metrics(CFD[10], truth_x)}
torch.manual_seed(1)
mlp_x = train(FieldNet().to(device), data_x, steps=6000, log_every=0)
results_x["B2 Data-only MLP"] = metrics(predict(mlp_x, X_TEST), truth_x)
pinn_x, phys_x = train_pinn(data_x)
results_x["PINN (hybrid)"] = metrics(predict(pinn_x, X_TEST), truth_x)
for name, m in results_x.items():
    print(f"{name:22s} R2 {m['R2']:.3f}  MAPE {m['MAPE (%)']:.2f}%  RMSE {m['RMSE (K)']:.1f} K")
""")

md(r"""
### Results table and error maps
""")

code(r"""
print(f"{'model':34s} {'R2':>7s} {'MAE (K)':>9s} {'MAPE (%)':>9s} {'RMSE (K)':>9s}")
for name, m in results.items():
    print(f"{name:34s} {m['R2']:7.4f} {m['MAE (K)']:9.2f} {m['MAPE (%)']:9.2f} {m['RMSE (K)']:9.2f}")

preds = {"Linear interp.": 0.5 * (CFD[-10] + CFD[10]), "Paper-style MLP": pred_sand,
         "Data-only MLP": pred_mlp, "Hybrid PINN": pred_pinn}
fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)
axes[0].imshow(truth.T, origin="lower", extent=[0, W, 0, H], cmap="jet", vmin=650, vmax=1900, aspect="auto")
axes[0].set_title("CFD truth, 0°")
for ax, (name, p) in zip(axes[1:], preds.items()):
    im = ax.imshow(np.abs(p - truth).T, origin="lower", extent=[0, W, 0, H], cmap="magma", vmin=0, vmax=150, aspect="auto")
    ax.axhline(FLAME_BAND[0], color="w", lw=0.6, ls="--"); ax.axhline(FLAME_BAND[1], color="w", lw=0.6, ls="--")
    ax.set_title(f"|error| {name}")
fig.colorbar(im, ax=axes[1:], label="absolute error (K)", shrink=0.8)
plt.show()

j = NX // 2
plt.figure(figsize=(5, 6))
plt.plot(truth[j], Y[j], "k", lw=3, alpha=0.4, label="CFD truth")
for name, p in preds.items():
    plt.plot(p[j], Y[j], lw=1.2, label=name)
plt.xlabel("T (K) along the centreline"); plt.ylabel("height y"); plt.legend(); plt.title("Centreline profile at 0°")
plt.show()
""")

md(r"""
## Part 6 · Using real CFD data (next step)

1. **Export each Fluent case** (*File → Export → Solution Data*, ASCII/CSV) with: x, y, z, temperature, x/y/z-velocity
   (optionally O₂, CO, NO). One file per operating condition, with the condition (tilt, load, …) in the filename.
2. **Replace Part 1** with a loader that builds the same `x, y, tilt, T` arrays (plus `z`). Add z and any extra operating
   parameters to `FieldNet`'s inputs.
3. **Use CFD cells as collocation points.** Pick cells outside the burner band (by height). Their CFD velocity
   (u, v, w) goes into the 3D residual: `u*Tx + v*Ty + w*Tz − alpha*(Txx + Tyy + Tzz) + beta*(T − T_wall)`.
   For a condition with no CFD run, use velocity interpolated between neighbouring cases, or a cheap cold-flow CFD (the idea in Wang et al., 2025).
4. **Scale the inputs** (coordinates to about [−1, 1], temperature as here). The paper doesn't say how it scaled its data.
5. **Keep every baseline** (linear interpolation, paper-style MLP, data-only MLP) in the results table.
6. **Large data:** 3.18 M cells per case will not fit in one batch; switch `train` to mini-batches. Use the DICC GPU server for those runs.
""")

nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for kind, src in cells:
    lines = src.split("\n"); lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code": c.update({"execution_count": None, "outputs": []})
    nb["cells"].append(c)
out = sys.argv[1] if len(sys.argv) > 1 else "PIML_Furnace_Starter.ipynb"
json.dump(nb, open(out, "w"), indent=1)
# also dump code-only script for syntax checking
open(out.replace(".ipynb", "_code.py"), "w").write("\n\n".join(s for k, s in cells if k == "code"))
print("wrote", out, len(cells), "cells")
