"""Builds PIML_Furnace_Master.ipynb v2: does the physics loss help? Train on 6 tilts, test on the unseen +20 deg image.
Run: python3 build_nb3.py [out.ipynb]"""
import json, sys

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# Does the physics loss help? Hybrid PINN vs the same network without physics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tansoonseng2710/fyp-piml-furnace/blob/main/PIML_Furnace_Master.ipynb)

**FYP 2026/2027 · KIG 4054 · model 1: hybrid PINN · test protocol agreed with the supervisor**

| | |
|---|---|
| **Question** | Does adding the physics loss improve the prediction, and by how much? |
| **Fair test** | One network (`FieldNet`), same inputs, same settings, same data, same number of steps. The **only** difference is the loss: `MSE` alone (data-only MLP) vs `MSE + 0.1 × physics` (hybrid PINN). |
| **Train** | 6 images: tilts −30°, −20°, −10°, 0°, +10°, +30° (10% of these points kept aside for validation) |
| **Test** | 1 image the network never sees: **+20°** |
| **Focus region** | The burner zone, x 0–14.5 m, y 9.7–32 m (the region marked by the supervisor) |
| **Main number** | R² on the +20° image, overall and in the burner zone, averaged over 3 seeds. Also MAE, MAPE, RMSE and a table of individual points. |

**Where the data comes from (please read).** The paper does not publish its CFD data ("available on request"). What it does publish is **Fig. 6**: 7 colour pictures of the CFD temperature on the furnace's central vertical plane, one per burner tilt.
This notebook's file is those 7 pictures **digitised**: every furnace pixel (0.12 m × 0.12 m) became one row `(x, y, tilt, T)`, with T read from the colour scale. About 72,000 pixels per picture × 7 = **506,706 rows**.
So the rows are **pixels of 7 CFD images**, not 506,706 separate CFD results: neighbouring pixels are nearly identical, and T is only known to the nearest colour band (83.5 K wide, so even a perfect model has MAE ≈ 21 K). The paper's own CFD has 3,183,898 cells per case in 3-D; none of those are available.

**Run it:** Colab → *Runtime → Change runtime type → T4 GPU* → *Run all*. About 10 minutes. The dataset downloads itself.
""")

code(r"""
import os, math, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch, torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "no GPU (Runtime > Change runtime type > T4 GPU)")

QUICK = False                    # True = short run to check everything works (numbers not meaningful)
TEST_TILT = 20                   # the one image held out for testing
FOCUS = dict(name="burner zone", x0=0.0, x1=14.6, y0=9.7, y1=32.0)     # region marked by the supervisor
CFG = dict(
    w_pde_list=[0.0, 0.1],       # 0.0 = data-only MLP, 0.1 = hybrid PINN. Same network either way.
    seeds=[0] if QUICK else [0, 1, 2],
    steps=1500 if QUICK else 6000,
    warm=400 if QUICK else 1500, # data-only steps before the physics loss is switched on (both models get identical steps)
    batch=16384, n_col=2048, lr=2e-3,
    width=128, depth=5, n_fourier=48, sigma=2.0,
    val_frac=0.10, log_every=500,
)
if DEVICE == "cpu" and not QUICK:  # no free GPU: a lighter run that still finishes in ~30 min on CPU
    CFG.update(seeds=[0], steps=3000, warm=750, batch=8192)
    print("CPU-lite run: 1 seed, 3000 steps, batch 8192 (GPU run uses 3 seeds, 6000 steps, batch 16384)")
""")

md(r"""
## 1 · Load the data and split it: 6 images train, +20° tests
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
    from google.colab import files
    print("Choose furnace_master_dataset.csv.gz ...")
    FNAME = list(files.upload().keys())[0]

D = pd.read_csv(FNAME)
D = D[~D.is_duplicate_panel.astype(bool)].reset_index(drop=True)       # Fig. 6 shows 0 deg twice; the repeat was only used to check the digitising
rng = np.random.default_rng(0)
D["split"] = np.where(D.tilt_deg == TEST_TILT, "test", np.where(rng.random(len(D)) < CFG["val_frac"], "val", "train"))
D["focus"] = D.x_m.between(FOCUS["x0"], FOCUS["x1"]) & D.y_m.between(FOCUS["y0"], FOCUS["y1"])

print(f"{len(D):,} rows from {D.tilt_deg.nunique()} images (tilts {sorted(D.tilt_deg.unique())})")
print(D.groupby("split").size().to_dict(), "| test = the whole", f"{TEST_TILT:+d}° image")
print(f"focus region ({FOCUS['name']}) in the test image: {int((D.focus & (D.split == 'test')).sum()):,} points")
""")

md(r"""
**What the 506,706 rows really are.** The table below answers the supervisor's question "how much data do we really have?".
The information content is 7 pictures. Thinning to one point per 1 m × 1 m square gives about 8,000 points in total, which is closer to the real amount of independent information.
""")

code(r"""
px = 0.1212                                                     # metres per pixel (58.3 m boiler height = 481 px)
info = D.groupby("tilt_deg").agg(pixels=("T_K", "size"), T_min=("T_K", "min"), T_max=("T_K", "max"))
info["points on a 1 m grid"] = D.assign(gx=(D.x_m // 1).astype(int), gy=(D.y_m // 1).astype(int)).groupby("tilt_deg").apply(lambda g: g[["gx", "gy"]].drop_duplicates().shape[0])
info["role"] = np.where(info.index == TEST_TILT, "TEST", "train + val")
display(info)
print(f"pixel size {px} m | colour bands: {D.band.nunique()} used, {(D.T_hi_K - D.T_lo_K).mean():.1f} K wide | "
      f"MAE floor for a perfect smooth model ≈ {(D.T_hi_K - D.T_lo_K).mean()/4:.0f} K")
print("Digitising check: Fig. 6 shows 0° twice; reading both copies gave the same colour band for 92.9% of pixels (mean difference 6 K).")
""")

code(r"""
fig, ax = plt.subplots(1, 7, figsize=(18, 5.6), sharey=True)
for a, t in zip(ax, sorted(D.tilt_deg.unique())):
    g = D[D.tilt_deg == t]
    sc = a.scatter(g.x_m, g.y_m, c=g.T_K, s=0.3, cmap="inferno", vmin=600, vmax=1900)
    a.add_patch(Rectangle((FOCUS["x0"], FOCUS["y0"]), FOCUS["x1"] - FOCUS["x0"], FOCUS["y1"] - FOCUS["y0"], fill=False, ec="cyan", lw=1.2, ls="--"))
    a.set_title(f"{t:+d}°" + ("  TEST" if t == TEST_TILT else ""), color="crimson" if t == TEST_TILT else "black")
    a.set_aspect("equal"); a.set_xlabel("x (m)")
ax[0].set_ylabel("height y (m)"); plt.colorbar(sc, ax=ax, label="CFD temperature (K)", shrink=0.7)
plt.suptitle("Train on 6 images, test on +20°. Dashed box = burner zone (focus region)"); plt.show()
""")

md(r"""
## 2 · Metrics and a simple reference

Every model is scored on the whole +20° image and on the burner zone inside it. R² = 1 is perfect; R² = 0 is no better than predicting the average temperature.
The reference is k-nearest-neighbours on `(x, y, tilt)`: it predicts a +20° pixel from the same spot at the neighbouring tilts (+10° and +30°). A neural model should beat it.
""")

code(r"""
def metrics(p, t):
    e = p - t
    return dict(R2=float(1 - (e**2).sum() / ((t - t.mean())**2).sum()), MAE=float(np.abs(e).mean()),
                MAPE=float(100 * np.abs(e / t).mean()), RMSE=float(np.sqrt((e**2).mean())))

REGIONS = {"whole +20° image": lambda d: np.ones(len(d), bool), f"{FOCUS['name']} (focus)": lambda d: d.focus.values,
           "upper shaft (physics)": lambda d: (d.zone == "shaft_upper").values, "back pass (physics)": lambda d: (d.zone == "back_pass").values}

def score(p, te, model, seed):
    rows = []
    for reg, f in REGIONS.items():
        m = f(te)
        rows.append(dict(model=model, seed=seed, region=reg, n=int(m.sum()), **metrics(p[m], te.T_K.values[m])))
    return rows

TR, VA, TE = D[D.split == "train"], D[D.split == "val"], D[D.split == "test"].reset_index(drop=True)
feat = lambda d: np.c_[d.x_m.values, d.y_m.values, d.tilt_deg.values * 0.3]       # 0.3 m per degree: neighbouring tilts (10°) count as 3 m apart
P_KNN = KNeighborsRegressor(5, weights="distance").fit(feat(TR), TR.T_K.values).predict(feat(TE))
ROWS = score(P_KNN, TE, "kNN reference", 0)
pd.DataFrame(ROWS).round(3)
""")

md(r"""
## 3 · The network and the physics loss

**Network** `FieldNet` (identical for both models): `(x, y, tilt)` → 48 sin + 48 cos Fourier features of `(x, y)` → 5 × 128 `tanh` layers → temperature.

**Physics loss** (hybrid only): a steady energy balance, checked at random "collocation" points where no temperature is needed:

| Region | Gas flow | Residual pushed to 0 |
|---|---|---|
| upper shaft, y 32–43 m | up, `u = C·T` | `C·T·∂T/∂y − α∇²T + 30·k·((T/1000)⁴ − 0.7⁴)` (radiation to the walls) |
| back pass | down | `−C·T·∂T/∂y − α∇²T + β·(T − T_sink)` (heat taken by the tube banks) |

`C` comes from the paper's Table 2 (530 kg/s of flue gas, ≈ 10 m/s). α, k, β, T_sink are learned. Collocation points use random tilts between −30° and +30°, so the physics is also applied at +20°, where there is no training data.

**Important for reading the burner-zone result:** the physics loss is **not** applied inside the burner zone (combustion is too complex for this simple balance). Any change there is indirect: the same network has to satisfy the physics just above the burners.
""")

code(r"""
X_MAX, Y_MAX, T_MEAN, T_STD = 33.0, 58.3, 1100.0, 450.0
T_WALL = 700.0
MDOT = 110.07 + 260.54 + 111.182 + 193.5 / 3.6 * (1 - 0.0984)       # kg/s flue gas, Table 2 (PA + SA + OFA + coal less ash)
def C_of(area): return MDOT * 8.314 / (101325.0 * 0.0295 * area)    # u = C*T  [m/s per K]
C_SHAFT, C_BACK = C_of(14.63 * 14.63), C_of(14.63 * 12.7)

class FieldNet(nn.Module):
    def __init__(self, width=128, depth=5, n_fourier=48, sigma=2.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(2, n_fourier, generator=g) * sigma)
        layers, d = [], 3 + 2 * n_fourier
        for _ in range(depth): layers += [nn.Linear(d, width), nn.Tanh()]; d = width
        layers.append(nn.Linear(d, 1)); self.net = nn.Sequential(*layers)
    def forward(self, x, y, t):
        xn, yn, tn = 2 * x / X_MAX - 1, 2 * y / Y_MAX - 1, t / 30.0
        p = 2 * math.pi * (torch.stack([xn, yn], -1) @ self.B)
        h = torch.cat([torch.stack([xn, yn, tn], -1), torch.sin(p), torch.cos(p)], -1)
        return T_MEAN + T_STD * self.net(h).squeeze(-1)

class HybridPhysics(nn.Module):
    def __init__(self, D, n_col):
        super().__init__()
        t0 = D[D.tilt_deg == 0]                     # geometry only (coordinates, no temperatures): no test information used
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
        return (Rs / 100.0).pow(2).mean() + (Rb / 100.0).pow(2).mean()

def to_t(a): return torch.tensor(a, dtype=torch.float32, device=DEVICE)

@torch.no_grad()
def predict(model, df, chunk=100_000):
    out = []
    for i in range(0, len(df), chunk):
        s = df.iloc[i:i + chunk]
        out.append(model(to_t(s.x_m.values), to_t(s.y_m.values), to_t(s.tilt_deg.values.astype(float))).cpu().numpy())
    return np.concatenate(out)
""")

md(r"""
## 4 · Train both models (3 seeds each)

Same seed = same starting weights and same data order for both models, so each pair differs only by the physics loss.
""")

code(r"""
def train(w_pde, seed, cfg):
    torch.manual_seed(seed); np.random.seed(seed)
    x, y, t, T = (to_t(TR[c].values.astype(float)) for c in ["x_m", "y_m", "tilt_deg", "T_K"])
    model = FieldNet(cfg["width"], cfg["depth"], cfg["n_fourier"], cfg["sigma"], seed).to(DEVICE)
    phys = HybridPhysics(D, cfg["n_col"]).to(DEVICE) if w_pde > 0 else None
    params = list(model.parameters()) + (list(phys.parameters()) if phys is not None else [])
    opt = torch.optim.Adam(params, lr=cfg["lr"]); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"])
    gen = torch.Generator(device=DEVICE).manual_seed(seed)                 # same data order for both models
    hist, t0, N = [], time.time(), len(x)
    for it in range(1, cfg["steps"] + 1):
        idx = torch.randint(0, N, (cfg["batch"],), device=DEVICE, generator=gen)
        loss_d = ((model(x[idx], y[idx], t[idx]) - T[idx]) / T_STD).pow(2).mean()
        loss_p, w = torch.zeros((), device=DEVICE), 0.0
        if phys is not None and it > cfg["warm"]:
            w = w_pde * min(1.0, (it - cfg["warm"]) / (0.2 * cfg["steps"]))
            loss_p = phys.loss(model)
        loss = loss_d + w * loss_p
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % cfg["log_every"] == 0 or it == cfg["steps"]:
            val_mae = float(np.abs(predict(model, VA) - VA.T_K.values).mean())
            hist.append((it, loss_d.item(), loss_p.item(), val_mae))
            print(f"  [{'MLP' if w_pde == 0 else 'PINN'} seed {seed}] step {it:5d}  data {loss_d.item():.4f}  physics {loss_p.item():.4f}  val MAE {val_mae:5.1f} K  ({time.time()-t0:.0f}s)")
    return model, phys, hist

NAME = {0.0: "MLP (data only)"}
MODELS, PRED, COEF = {}, {}, []
for seed in CFG["seeds"]:
    for w_pde in CFG["w_pde_list"]:
        name = NAME.get(w_pde, f"Hybrid PINN (w={w_pde})")
        model, phys, hist = train(w_pde, seed, CFG)
        p = predict(model, TE); PRED[(name, seed)] = p; MODELS[(name, seed)] = (model, phys, hist)
        ROWS += score(p, TE, name, seed)
        if phys is not None: COEF.append(dict(seed=seed, **phys.coeffs()))
print("done")
""")

md(r"""
## 5 · Results: does the physics loss help?
""")

code(r"""
R = pd.DataFrame(ROWS)
order = ["kNN reference", "MLP (data only)"] + [n for n in R.model.unique() if n.startswith("Hybrid")]
agg = R.groupby(["region", "model"], sort=False)[["R2", "MAE", "MAPE", "RMSE"]].agg(["mean", "std"])
tab = pd.DataFrame({c: agg[(c, "mean")].map(("{:.4f}" if c == "R2" else "{:.1f}").format) + " ± " + agg[(c, "std")].fillna(0).map(("{:.4f}" if c == "R2" else "{:.1f}").format)
                    for c in ["R2", "MAE", "MAPE", "RMSE"]})
tab.columns = ["R²", "MAE (K)", "MAPE (%)", "RMSE (K)"]
print(f"Test = unseen {TEST_TILT:+d}° image. Mean ± spread over {len(CFG['seeds'])} seed(s).")
display(tab)
""")

code(r"""
mlp, pinn = "MLP (data only)", [n for n in R.model.unique() if n.startswith("Hybrid")][0]
rows = []
for reg in REGIONS:
    a = R[(R.region == reg) & (R.model == mlp)].set_index("seed"); b = R[(R.region == reg) & (R.model == pinn)].set_index("seed")
    dR2 = b.R2 - a.R2; dMAE = 100 * (a.MAE - b.MAE) / a.MAE
    rows.append({"region": reg, "R² data-only": a.R2.mean(), "R² hybrid": b.R2.mean(), "R² gain (points)": dR2.mean(),
                 "MAE data-only (K)": a.MAE.mean(), "MAE hybrid (K)": b.MAE.mean(), "MAE reduction (%)": dMAE.mean(),
                 "hybrid better in": f"{int((dR2 > 0).sum())} of {len(dR2)} seeds"})
IMPROVE = pd.DataFrame(rows).set_index("region")
print("The answer in one table: hybrid PINN minus data-only MLP (same network, only the loss differs)")
display(IMPROVE.round({"R² data-only": 4, "R² hybrid": 4, "R² gain (points)": 4, "MAE data-only (K)": 1, "MAE hybrid (K)": 1, "MAE reduction (%)": 1}))
f = IMPROVE.loc[f"{FOCUS['name']} (focus)"]
print(f"\nBurner zone: R² {f['R² data-only']:.4f} → {f['R² hybrid']:.4f}  ({f['R² gain (points)']:+.4f}),  "
      f"MAE {f['MAE data-only (K)']:.1f} → {f['MAE hybrid (K)']:.1f} K  ({f['MAE reduction (%)']:+.1f}% error reduction), hybrid better in {f['hybrid better in']}.")
""")

md(r"""
### Point by point in the burner zone

15 fixed points spread over the focus region of the +20° image: the CFD value, what each model predicts there (mean over seeds), and the error.
""")

code(r"""
pts = []
for yy in [12, 17, 22, 27, 31]:
    for xx in [3.5, 7.3, 11.0]:
        d = (TE.x_m - xx) ** 2 + (TE.y_m - yy) ** 2; i = int(d.idxmin())
        pm = np.mean([PRED[(mlp, s)][i] for s in CFG["seeds"]]); pp = np.mean([PRED[(pinn, s)][i] for s in CFG["seeds"]])
        cfd = TE.T_K[i]
        pts.append({"x (m)": round(TE.x_m[i], 1), "y (m)": round(TE.y_m[i], 1), "CFD T (K)": cfd, "data-only T (K)": pm, "hybrid T (K)": pp,
                    "data-only error (K)": abs(pm - cfd), "hybrid error (K)": abs(pp - cfd), "closer": "hybrid" if abs(pp - cfd) < abs(pm - cfd) else "data-only"})
PTS = pd.DataFrame(pts)
display(PTS.round(0).astype({"closer": str}))
print(f"hybrid closer at {int((PTS.closer == 'hybrid').sum())} of {len(PTS)} points | mean point error: data-only {PTS['data-only error (K)'].mean():.0f} K, hybrid {PTS['hybrid error (K)'].mean():.0f} K")
""")

code(r"""
s0 = CFG["seeds"][0]
fig, ax = plt.subplots(1, 5, figsize=(19, 7), sharey=True)
panels = [("CFD (truth)", TE.T_K.values, "inferno", (600, 1900)), ("data-only MLP", PRED[(mlp, s0)], "inferno", (600, 1900)),
          ("hybrid PINN", PRED[(pinn, s0)], "inferno", (600, 1900)),
          ("|error| data-only", np.abs(PRED[(mlp, s0)] - TE.T_K.values), "magma_r", (0, 250)), ("|error| hybrid", np.abs(PRED[(pinn, s0)] - TE.T_K.values), "magma_r", (0, 250))]
for a, (title, v, cm, (lo, hi)) in zip(ax, panels):
    sc = a.scatter(TE.x_m, TE.y_m, c=v, s=0.3, cmap=cm, vmin=lo, vmax=hi)
    a.add_patch(Rectangle((FOCUS["x0"], FOCUS["y0"]), FOCUS["x1"] - FOCUS["x0"], FOCUS["y1"] - FOCUS["y0"], fill=False, ec="cyan", lw=1.2, ls="--"))
    a.set_title(title); a.set_aspect("equal"); a.set_xlabel("x (m)"); plt.colorbar(sc, ax=a, shrink=0.5)
ax[0].set_ylabel("height y (m)"); plt.suptitle(f"Unseen {TEST_TILT:+d}° image, seed {s0}. Dashed box = burner zone"); plt.show()

z = TE.focus.values
fig, ax = plt.subplots(1, 3, figsize=(11, 6), sharey=True)
for a, (title, v) in zip(ax, [("CFD", TE.T_K.values), ("data-only MLP", PRED[(mlp, s0)]), ("hybrid PINN", PRED[(pinn, s0)])]):
    sc = a.scatter(TE.x_m[z], TE.y_m[z], c=v[z], s=1.5, cmap="inferno", vmin=600, vmax=1900)
    a.set_title(f"{title}\nburner zone R² {metrics(v[z], TE.T_K.values[z])['R2']:.3f}" if title != "CFD" else "CFD (truth)"); a.set_aspect("equal"); a.set_xlabel("x (m)")
ax[0].set_ylabel("y (m)"); plt.colorbar(sc, ax=ax, shrink=0.7, label="T (K)"); plt.show()
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for (name, seed), (m, ph, h) in MODELS.items():
    h = np.array(h); ax[0].plot(h[:, 0], h[:, 3], color="tab:blue" if name == mlp else "tab:red", alpha=0.7, label=f"{name}, seed {seed}")
    if ph is not None: ax[1].semilogy(h[:, 0], np.maximum(h[:, 2], 1e-6), color="tab:red", alpha=0.7)
ax[0].set_title("validation MAE (K), 10% of training-image points"); ax[0].set_xlabel("step"); ax[0].legend(fontsize=7)
ax[1].set_title("physics loss (switched on after warm-up)"); ax[1].set_xlabel("step"); plt.tight_layout(); plt.show()
if COEF:
    print("Learned physics coefficients (a-priori: alpha 1-10 m²/s, k_rad ≈ 1, beta ≈ 1-2 1/s, T_sink ≈ 740 K)")
    display(pd.DataFrame(COEF).round(3))
R.to_csv("results_tilt20.csv", index=False); IMPROVE.to_csv("improvement_tilt20.csv"); PTS.to_csv("points_tilt20.csv", index=False)
print("saved results_tilt20.csv, improvement_tilt20.csv, points_tilt20.csv")
""")

md(r"""
### Export: predicted maps of the +20° image (for slides)

Mean prediction over the seeds of each model on a 0.5 m grid, printed as text (1 byte per cell, (T − 500 K) / 8; 0 = outside the furnace). The slides redraw these next to the CFD map.
""")

code(r"""
import base64
G = 0.5; NX, NY = int(np.ceil(X_MAX / G)), int(np.ceil(Y_MAX / G))
gx = np.clip((TE.x_m.values // G).astype(int), 0, NX - 1); gy = np.clip((TE.y_m.values // G).astype(int), 0, NY - 1)
cells = np.unique(np.c_[gx, gy], axis=0)
cdf = pd.DataFrame({"x_m": (cells[:, 0] + 0.5) * G, "y_m": (cells[:, 1] + 0.5) * G, "tilt_deg": TEST_TILT})
print("MAPEXPORT_BEGIN")
print(f"grid {G} {NX} {NY} tilt {TEST_TILT} seeds {len(CFG['seeds'])}")
for name in [mlp, pinn]:
    p = np.mean([predict(MODELS[(name, s)][0], cdf) for s in CFG["seeds"]], axis=0)
    a = np.zeros((NY, NX), np.uint8)
    a[cells[:, 1], cells[:, 0]] = np.clip(np.round((p - 500) / 8), 1, 255).astype(np.uint8)
    print("MAP", name.replace(" ", "_"), base64.b64encode(a.tobytes()).decode())
print("TABLE"); print(IMPROVE.round(4).to_string())
print("POINTS hybrid closer", int((PTS.closer == "hybrid").sum()), "of", len(PTS), "| mean point error data-only", round(PTS["data-only error (K)"].mean(), 1), "hybrid", round(PTS["hybrid error (K)"].mean(), 1))
print("MAPEXPORT_END")
""")

md(r"""
## 6 · How to read the answer

* **The key table is "hybrid PINN minus data-only MLP".** R² gain in points (for example 0.95 → 0.97 is +0.02) and the MAE reduction in %. Because the two models share the network, data, steps and seeds, the difference is caused by the physics loss.
* **Is it real or luck?** Look at "hybrid better in N of 3 seeds" and the ± spread. A gain smaller than the spread between seeds is not a reliable improvement.
* **Burner zone vs physics zones.** The physics is applied in the upper shaft and back pass. If the gain is large there but small in the burner zone, the physics works where it is applied but does not yet reach the flame region. The next step would then be a burner-zone term (needs a heat-release model).
* **Floor.** Even a perfect model has MAE ≈ 21 K against these colour-band labels, so improvements near that level are at the limit of what this data can show.
* **Not comparable with the paper's Table 4.** The paper uses continuous 3-D CFD and 18 inputs (velocity, O₂, CO …); this uses 7 digitised 2-D pictures and 3 inputs `(x, y, tilt)`.
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
