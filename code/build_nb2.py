"""Builds PIML_Furnace_Master.ipynb (no nbformat needed). Run: python3 build_nb2.py out.ipynb"""
import json, sys

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# Hybrid PINN on the master furnace dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tansoonseng2710/fyp-piml-furnace/blob/main/PIML_Furnace_Master.ipynb)

**FYP 2026/2027 · KIG 4054 · model 1: hybrid PINN**

**Data:** `furnace_master_dataset.csv.gz` = Fig. 6 of Xue et al. (2024, *Energy* 286, 129568) digitised: the CFD temperature
on the furnace central vertical plane for 7 burner tilts (−30° … +30°), 506,706 points. It is everything numeric the paper
gives; the paper's raw 3-D CFD cells are "available on request".

**What this notebook does**

| Step | |
|---|---|
| 1 | Load the master file, look at the 7 fields |
| 2 | **Train on 9/10, test on 1/10**, two ways: `split_random` (random points, the 9/10 : 1/10 you asked for) and `split_block` (4 m × 4 m regions held out for every tilt, the honest version). A third protocol, `split_tilt0`, holds out the whole 0° tilt (the paper's own test, about 14%) |
| 3 | Baselines: k-nearest-neighbours (same tilt, and across tilts), data-only MLP |
| 4 | **Hybrid PINN**: same network + a steady energy-balance residual (convection – diffusion – heat loss) where the burners are not active, with the diffusivity and loss coefficients *learned* |
| 5 | Results table in the paper's four metrics (R², MAE, MAPE, RMSE), per-zone errors, error maps, learned physics coefficients |

**Three things to know before reading any number**

1. **Temperatures are quantised.** Fig. 6 has 20 colour bands, 83.5 K wide. Even a perfect smooth model has MAE ≈ 21 K against these labels.
   So results are *not* comparable with the paper's Table 4 (MAE 36–41 K on continuous 3-D CFD at an unseen tilt).
2. **`split_random` is optimistic.** Neighbouring pixels are almost identical, so the test points are all next to training points. The table shows this on purpose: kNN does well on it.
3. **One plane, one seed.** This is the first draft. It checks that the pipeline and the physics term work. Do not quote the numbers as final.

**Run it:** Colab → *Runtime → Change runtime type → T4 GPU* → *Run all*. The dataset downloads itself from the project's GitHub repo. Set `QUICK = True` for a short smoke test first.
""")

code(r"""
import os, math, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "no GPU (Runtime > Change runtime type > T4 GPU)")

QUICK = False            # True = 1,500 steps, for a quick check that everything runs
CFG = dict(
    protocols=["split_random", "split_block", "split_tilt0"],   # split_tilt0 = whole 0° tilt unseen, the paper's own test (~14% test, not 1/10)
    w_pde_list=[0.0, 0.1],                       # 0.0 = data-only MLP baseline; 0.1 = hybrid PINN. Try [0.0, 0.01, 0.1, 1.0]
    seeds=[0],                                   # use [0, 1, 2] for mean ± spread
    steps=1500 if QUICK else 6000,
    warm=400 if QUICK else 1500,                 # data-only steps before the physics term is switched on
    batch=16384, n_col=2048, lr=2e-3,
    width=128, depth=5, n_fourier=48, sigma=2.0,
    log_every=500,
)
""")

md(r"""
## 1 · Load the master file

The file downloads automatically from the project's GitHub repo. If that fails (no internet), you are asked to upload `furnace_master_dataset.csv.gz` instead.
""")

code(r"""
FNAME = "furnace_master_dataset.csv.gz"
DATA_URL = "https://raw.githubusercontent.com/tansoonseng2710/fyp-piml-furnace/main/data/furnace_master_dataset.csv.gz"
if not os.path.exists(FNAME):
    try:
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, FNAME)
        print("downloaded", FNAME, f"({os.path.getsize(FNAME)/1e6:.1f} MB)")
    except Exception as e:
        print("download failed:", e)
if not os.path.exists(FNAME):
    try:
        from google.colab import files
        print("Choose furnace_master_dataset.csv.gz ...")
        up = files.upload()
        FNAME = list(up.keys())[0]
    except ImportError:
        raise FileNotFoundError("Put furnace_master_dataset.csv.gz next to this notebook.")

D = pd.read_csv(FNAME)
D = D[~D.is_duplicate_panel].reset_index(drop=True)          # panel 5 is a repeat of 0 deg; it was only used for the QA check
D["split_tilt0"] = np.where(D.tilt_deg == 0, "test", "train")  # optional paper-style protocol: whole 0 deg tilt unseen

print(f"{len(D):,} rows | tilts {sorted(D.tilt_deg.unique())} | T from {D.T_K.min():.0f} to {D.T_K.max():.0f} K")
print("MAE floor from colour quantisation (perfect smooth model vs band-midpoint labels): about %.0f K" % ((D.T_hi_K - D.T_lo_K).mean() / 4))
for c in ["split_random", "split_block"]:
    print(c, D[c].value_counts().to_dict())
print(D.groupby("zone").size().to_dict())
D.head()
""")

code(r"""
fig, ax = plt.subplots(1, 7, figsize=(18, 5.5), sharey=True)
for a, t in zip(ax, sorted(D.tilt_deg.unique())):
    g = D[D.tilt_deg == t]
    sc = a.scatter(g.x_m, g.y_m, c=g.T_K, s=0.4, cmap="jet", vmin=330, vmax=2000)
    a.set_title(f"{t:+d}°"); a.set_aspect("equal"); a.set_xlabel("x (m)")
ax[0].set_ylabel("height y (m)"); plt.colorbar(sc, ax=ax, label="T (K)", shrink=0.7); plt.show()
""")

md(r"""
## 2 · Metrics and the baselines

The paper's four metrics. The kNN baselines predict every test pixel from the 5 nearest *training* pixels, either of the same tilt or across tilts.
On `split_random` they look excellent, and that is the warning: the test is too easy. On `split_block` they only have to bridge a 4 m hole in a smooth field, so they stay strong too.
**kNN is a hard baseline here. The neural models have to beat it to justify themselves**, and the place they can is the unseen tilt (`split_tilt0`), where the same-tilt kNN cannot run at all.
""")

code(r"""
def metrics(p, t):
    e = p - t
    return dict(R2=float(1 - (e**2).sum() / ((t - t.mean())**2).sum()), MAE=float(np.abs(e).mean()),
                MAPE=float(100 * np.abs(e / t).mean()), RMSE=float(np.sqrt((e**2).mean())))

ZONES = ["shaft_upper", "back_pass", "burner_zone", "hopper", "horizontal_pass", "nose"]
def zone_mae(p, df):
    return {f"MAE {z}": (float(np.abs(p[(df.zone.values == z)] - df.T_K.values[(df.zone.values == z)]).mean()) if (df.zone.values == z).any() else np.nan) for z in ZONES}

def knn_baseline(split_col, k=5, tilt_scale=None):
    # tilt_scale=None: neighbours only from the same tilt. Otherwise one kNN on (x, y, tilt*tilt_scale), so an unseen tilt gets neighbours from adjacent tilts.
    P = np.full(len(D), np.nan)
    if tilt_scale is None:
        for t, g in D.groupby("tilt_deg"):
            tr, te = g[g[split_col] == "train"], g[g[split_col] == "test"]
            if len(te) == 0 or len(tr) == 0: continue
            P[te.index] = KNeighborsRegressor(k, weights="distance").fit(tr[["x_m", "y_m"]].values, tr.T_K.values).predict(te[["x_m", "y_m"]].values)
    else:
        feat = lambda d: np.c_[d.x_m.values, d.y_m.values, d.tilt_deg.values * tilt_scale]
        tr, te = D[D[split_col] == "train"], D[D[split_col] == "test"]
        P[te.index] = KNeighborsRegressor(k, weights="distance").fit(feat(tr), tr.T_K.values).predict(feat(te))
    return P

RESULTS = []
for proto in CFG["protocols"]:
    for name, kw in [("kNN (same tilt)", dict()), ("kNN (x, y, tilt)", dict(tilt_scale=0.3))]:    # 0.3 m per degree: adjacent tilts (10°) count as 3 m apart
        P = knn_baseline(proto, **kw); te = D[(D[proto] == "test") & ~np.isnan(P)]
        if len(te) == 0: continue                                                                   # same-tilt kNN cannot predict an unseen tilt
        RESULTS.append(dict(protocol=proto, model=name, seed=0, n_test=len(te), **metrics(P[te.index], te.T_K.values), **zone_mae(P[te.index], te)))
pd.DataFrame(RESULTS).round(3)
""")

md(r"""
## 3 · The hybrid PINN

**Network** `FieldNet`: input `(x, y, tilt)` → Fourier features of `(x, y)` → 5 × 128 `tanh` layers → temperature. `tanh` (not ReLU) because the physics term needs second derivatives and ReLU's second derivative is zero.

**Physics** `HybridPhysics`: a steady energy balance, switched on only where the burners are no longer releasing heat, in two regions where the flow direction is known and simple:

| Region | Flow | Equation (residual to drive to 0) |
|---|---|---|
| upper shaft, y 32–43 m | up, `u = C·T` | `C·T·T_y − α∇²T + 30·k·((T/1000)⁴ − (700/1000)⁴)`  (radiation to the water walls) |
| back pass | down, `u = C·T` | `−C·T·T_y − α∇²T + β·(T − T_sink)`  (heat taken by the tube banks) |

* `C = ṁ R / (p M A)` comes from the paper's **Table 2**: ṁ = 530 kg/s of flue gas (primary + secondary + over-fire air + coal − ash), ideal-gas density, cross-section 14.63 m × width. It gives ≈ 10 m/s at 1,500 K, a realistic furnace velocity.
* **Learned**: α (turbulent diffusivity, m²/s) in each region, `k` (radiation scale, 1 ≈ textbook), β (1/s) and `T_sink`. Nothing else is tuned.
* Collocation points are placed on the real geometry at **random tilts between −30° and +30°**, so the physics is also enforced at tilts with no data.
* Training: data-only for `warm` steps, then the physics weight ramps up to `w_pde` (0.1). Both models use identical data, steps and schedule.

**Honest limits of this physics** (from checking it against the data): the radiative balance fits the first ~4 m above the burners (implied k ≈ 1.3–1.7) but the CFD temperature then plateaus near 1,560–1,600 K, so the shaft term is only a soft prior.
The flow directions are assumed, not taken from CFD velocity. Both are why the weight is small and the coefficients are learned. The next version should use CFD velocity.
""")

code(r"""
X_MAX, Y_MAX, T_MEAN, T_STD = 33.0, 58.3, 1100.0, 450.0
T_WALL = 700.0                                                      # water-wall temperature, K (assumption)
MDOT = 110.07 + 260.54 + 111.182 + 193.5 / 3.6 * (1 - 0.0984)       # kg/s flue gas, Table 2 (PA + SA + OFA + coal less ash)
def C_of(area): return MDOT * 8.314 / (101325.0 * 0.0295 * area)    # u = C*T  [m/s per K]
C_SHAFT, C_BACK = C_of(14.63 * 14.63), C_of(14.63 * 12.7)
print(f"gas flow {MDOT:.0f} kg/s | u(1500 K) in the shaft = {C_SHAFT*1500:.1f} m/s | u(800 K) in the back pass = {C_BACK*800:.1f} m/s")

class FieldNet(nn.Module):
    def __init__(self, width=128, depth=5, n_fourier=48, sigma=2.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(2, n_fourier, generator=g) * sigma)
        layers, d = [], 3 + 2 * n_fourier
        for _ in range(depth): layers += [nn.Linear(d, width), nn.Tanh()]; d = width
        layers.append(nn.Linear(d, 1)); self.net = nn.Sequential(*layers)
    def forward(self, x, y, t):                    # x, y in metres, t in degrees -> temperature in K. Inputs scaled inside, so d/dx is per metre.
        xn, yn, tn = 2 * x / X_MAX - 1, 2 * y / Y_MAX - 1, t / 30.0
        p = 2 * math.pi * (torch.stack([xn, yn], -1) @ self.B)
        h = torch.cat([torch.stack([xn, yn, tn], -1), torch.sin(p), torch.cos(p)], -1)
        return T_MEAN + T_STD * self.net(h).squeeze(-1)

class HybridPhysics(nn.Module):
    def __init__(self, D, n_col):
        super().__init__()
        t0 = D[D.tilt_deg == 0]                     # coordinates only (no temperatures) -> no test leakage
        self.register_buffer("pool_s", torch.tensor(t0[t0.zone == "shaft_upper"][["x_m", "y_m"]].values, dtype=torch.float32))
        self.register_buffer("pool_b", torch.tensor(t0[t0.zone == "back_pass"][["x_m", "y_m"]].values, dtype=torch.float32))
        self.n_col = n_col
        self.raw = nn.Parameter(torch.zeros(5))     # log alpha_shaft, log k_rad, log alpha_back, log beta_back, sink offset
    def coeffs(self):
        r = self.raw.detach().cpu()
        return dict(alpha_shaft=float(r[0].exp()), k_rad=float(r[1].exp()), alpha_back=float(r[2].exp()), beta_back=float(r[3].exp()), T_sink=float(740 + 50 * r[4]))
    def _points(self, pool):
        i = torch.randint(0, len(pool), (self.n_col,), device=pool.device)
        return pool[i, 0], pool[i, 1], torch.rand(self.n_col, device=pool.device) * 60.0 - 30.0
    @staticmethod
    def _derivs(model, x, y, t):
        x = x.clone().requires_grad_(True); y = y.clone().requires_grad_(True)
        T = model(x, y, t)
        Tx, Ty = torch.autograd.grad(T.sum(), (x, y), create_graph=True)
        Txx = torch.autograd.grad(Tx.sum(), x, create_graph=True)[0]
        Tyy = torch.autograd.grad(Ty.sum(), y, create_graph=True)[0]
        return T, Ty, Txx + Tyy
    def loss(self, model):
        a_s, k, a_b, b_b = (self.raw[i].exp() for i in range(4)); T_sink = 740.0 + 50.0 * self.raw[4]
        x, y, t = self._points(self.pool_s); T, Ty, lap = self._derivs(model, x, y, t)
        Rs = C_SHAFT * T * Ty - a_s * lap + 30.0 * k * ((T / 1000.0) ** 4 - (T_WALL / 1000.0) ** 4)
        x, y, t = self._points(self.pool_b); T, Ty, lap = self._derivs(model, x, y, t)
        Rb = -C_BACK * T * Ty - a_b * lap + b_b * (T - T_sink)
        return (Rs / 100.0).pow(2).mean() + (Rb / 100.0).pow(2).mean()          # residuals in units of 100 K/s

@torch.no_grad()
def predict(model, df, chunk=100_000):
    out = []
    for i in range(0, len(df), chunk):
        s = df.iloc[i:i + chunk]
        out.append(model(*[torch.tensor(s[c].values, dtype=torch.float32, device=DEVICE) for c in ["x_m", "y_m", "tilt_deg"]]).cpu().numpy())
    return np.concatenate(out)
""")

code(r"""
def to_t(a): return torch.tensor(a, dtype=torch.float32, device=DEVICE)

def train(split_col, w_pde, seed, cfg):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = D[D[split_col] == "train"]
    x, y, t, T = (to_t(tr[c].values) for c in ["x_m", "y_m", "tilt_deg", "T_K"])
    model = FieldNet(cfg["width"], cfg["depth"], cfg["n_fourier"], cfg["sigma"], seed).to(DEVICE)
    phys = HybridPhysics(D, cfg["n_col"]).to(DEVICE) if w_pde > 0 else None
    params = list(model.parameters()) + (list(phys.parameters()) if phys is not None else [])
    opt = torch.optim.Adam(params, lr=cfg["lr"]); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"])
    hist, t0, N = [], time.time(), len(x)
    for it in range(1, cfg["steps"] + 1):
        idx = torch.randint(0, N, (cfg["batch"],), device=DEVICE)
        loss_d = ((model(x[idx], y[idx], t[idx]) - T[idx]) / T_STD).pow(2).mean()
        loss_p, w = torch.zeros((), device=DEVICE), 0.0
        if phys is not None and it > cfg["warm"]:
            w = w_pde * min(1.0, (it - cfg["warm"]) / (0.2 * cfg["steps"]))     # ramp the physics weight in
            loss_p = phys.loss(model)
        loss = loss_d + w * loss_p
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % cfg["log_every"] == 0 or it == cfg["steps"]:
            hist.append((it, loss_d.item(), loss_p.item()))
            print(f"  [{split_col} | w_pde={w_pde} | seed {seed}] step {it:5d}  data loss {loss_d.item():.4f}  physics loss {loss_p.item():.4f}  ({time.time()-t0:.0f}s)")
    return model, phys, hist

MODELS = {}
for proto in CFG["protocols"]:
    te = D[D[proto] == "test"]
    for w_pde in CFG["w_pde_list"]:
        for seed in CFG["seeds"]:
            model, phys, hist = train(proto, w_pde, seed, CFG)
            p = predict(model, te)
            name = "MLP (data only)" if w_pde == 0 else f"Hybrid PINN (w={w_pde})"
            row = dict(protocol=proto, model=name, seed=seed, n_test=len(te), **metrics(p, te.T_K.values), **zone_mae(p, te))
            if phys is not None: row.update({f"learned {k}": v for k, v in phys.coeffs().items()})
            RESULTS.append(row); MODELS[(proto, w_pde, seed)] = (model, phys, hist)
print("done")
""")

md(r"""
## 4 · Results
""")

code(r"""
R = pd.DataFrame(RESULTS)
cols = ["R2", "MAE", "MAPE", "RMSE"]
summary = R.groupby(["protocol", "model"], sort=False)[cols].mean().round(3)
print("Paper's metrics on the 1/10 test set (MAE and RMSE in K, MAPE in %)")
print("Reference: Xue et al. DNN, unseen 0° tilt, 3-D: R2 0.979, MAE 35.8, MAPE 3.61, RMSE 64.4  (different data: NOT comparable)\n")
display(summary)
zc = [c for c in R.columns if c.startswith("MAE ")]
print("\nMAE (K) by region. The physics acts only in shaft_upper and back_pass; the other regions are controls.")
display(R.groupby(["protocol", "model"], sort=False)[zc].mean().round(1))
""")

code(r"""
learned = R[[c for c in R.columns if c.startswith("learned")] + ["protocol", "model"]].dropna()
print("Learned physics coefficients  (a-priori: alpha 1-10 m²/s, k_rad ≈ 1, beta_back ≈ 1-2 1/s, T_sink ≈ 740 K)")
display(learned.round(3))
""")

code(r"""
# Error maps at 0° for the block split: where do the held-out 4 m blocks go wrong, and does physics help there?
proto = "split_block"
keys = [(proto, w, CFG["seeds"][0]) for w in CFG["w_pde_list"] if (proto, w, CFG["seeds"][0]) in MODELS]
if keys:
    g = D[D.tilt_deg == 0]; te = g[g[proto] == "test"]
    fig, ax = plt.subplots(1, len(keys) + 1, figsize=(4.2 * (len(keys) + 1), 7), sharey=True)
    ax[0].scatter(g.x_m, g.y_m, c="lightgray", s=0.3); sc = ax[0].scatter(te.x_m, te.y_m, c=te.T_K, s=0.5, cmap="jet", vmin=330, vmax=2000)
    ax[0].set_title("held-out blocks (truth)")
    for a, k in zip(ax[1:], keys):
        p = predict(MODELS[k][0], te); a.scatter(g.x_m, g.y_m, c="lightgray", s=0.3)
        im = a.scatter(te.x_m, te.y_m, c=p - te.T_K.values, s=0.5, cmap="RdBu_r", vmin=-250, vmax=250)
        a.set_title(("MLP" if k[1] == 0 else f"Hybrid PINN w={k[1]}") + f"\nMAE {np.abs(p - te.T_K.values).mean():.0f} K")
    for a in ax: a.set_aspect("equal")
    plt.colorbar(im, ax=ax[1:], label="prediction − truth (K)", shrink=0.6); plt.show()
""")

code(r"""
# Training curves
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for k, (m, ph, h) in MODELS.items():
    if k[2] != CFG["seeds"][0]: continue
    h = np.array(h); lab = f"{k[0]}, " + ("MLP" if k[1] == 0 else f"PINN w={k[1]}")
    ax[0].semilogy(h[:, 0], h[:, 1], label=lab)
    if k[1] > 0: ax[1].semilogy(h[:, 0], np.maximum(h[:, 2], 1e-6), label=lab)
ax[0].set_title("data loss (normalised MSE)"); ax[1].set_title("physics loss (switched on after warm-up)")
ax[0].legend(fontsize=7); ax[1].legend(fontsize=7); ax[0].set_xlabel("step"); ax[1].set_xlabel("step"); plt.tight_layout(); plt.show()

R.to_csv("piml_results.csv", index=False); print("saved piml_results.csv")
""")

md(r"""
## 5 · How to read the results

* **Compare `MLP` with `Hybrid PINN` on `split_block`**, and look at the `shaft_upper` and `back_pass` columns: that is the only place the physics acts. If the PINN is not better there, the physics term is not helping yet. That is a valid finding. Do not tune `w_pde` on this test set; carve a validation block out of the training blocks first.
* **`split_random` vs `split_block` vs `split_tilt0`**: the gap between them is the leakage. Report all three. `split_tilt0` (an unseen tilt) is the one that matches the paper's goal and the one where kNN cannot cheat from neighbouring pixels.
* **Beat kNN first.** If the MLP and PINN lose to `kNN (x, y, tilt)` on `split_tilt0`, a lookup table beats the network on this data, and the real data (continuous, 3-D, sparse) is the next test, not more tuning.
* **Against the quantisation floor (≈ 21 K MAE)**: a model within a few K of it has learned the figure, not necessarily the physics.
* **Learned coefficients** that stay at their start values (α = 1, k = 1, β = 1) mean the physics gradient is weak; ones that move far from the a-priori ranges mean the assumed flow or loss form is wrong.

## 6 · Next steps (in order)

1. **Real CFD data.** Email the paper's authors, or run your own Fluent cases (3 or more tilts). Then the labels are continuous (no 83.5 K bands), the velocity field `(u, v, w)` comes from CFD, and the physics can be the full 3-D balance with CFD velocity instead of the assumed `u = C·T`.
2. **Paper-faithful test:** add `"split_tilt0"` to `protocols` (holds out the whole 0° tilt; it is the paper's own test) and train on the other 6.
3. **Seeds and validation:** `seeds=[0,1,2]`; choose `w_pde` and the burner-zone height (`FLAME_TOP` = 32 m in the master file) on a validation block, then report once on the test set.
4. **Add the paper's baselines** (ELM, ESN, 18-input DNN) on the same split so the comparison table is complete.
5. **Sparse-sensor correction:** the paper has 6 plant sensors; add them as an extra data term (the starter notebook already shows how).
""")

nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for kind, src in cells:
    lines = src.split("\n"); lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code": c.update({"execution_count": None, "outputs": []})
    nb["cells"].append(c)
out = sys.argv[1] if len(sys.argv) > 1 else "PIML_Furnace_Master.ipynb"
json.dump(nb, open(out, "w"), indent=1)
open(out.replace(".ipynb", "_code.py"), "w").write("\n\n".join(s for k, s in cells if k == "code"))
print("wrote", out, len(cells), "cells")
