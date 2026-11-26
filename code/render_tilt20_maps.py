import sys, base64, re, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
txt = open(sys.argv[1]).read()
G, NX, NY = 0.5, 66, 117
maps = {}
for name, b in re.findall(r"MAP (\S+) ([A-Za-z0-9+/=]+)", txt):
    a = np.frombuffer(base64.b64decode(b), np.uint8).reshape(NY, NX).astype(float)
    maps[name] = np.where(a > 0, 500 + 8 * a, np.nan)
D = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "furnace_master_dataset.csv.gz"); t = D[(D.tilt_deg == 20) & (~D.is_duplicate_panel.astype(bool))]
gx = np.clip((t.x_m // G).astype(int), 0, NX - 1); gy = np.clip((t.y_m // G).astype(int), 0, NY - 1)
cfd = t.groupby([gy, gx]).T_K.mean(); C = np.full((NY, NX), np.nan)
for (yy, xx), v in cfd.items(): C[yy, xx] = v
mlp = [v for k, v in maps.items() if k.startswith("MLP")][0]; pinn = [v for k, v in maps.items() if k.startswith("Hybrid")][0]
m = ~np.isnan(C) & ~np.isnan(mlp) & ~np.isnan(pinn)
for A in (mlp, pinn): A[~m] = np.nan
C[~m] = np.nan
def r2(p, q, k): e = p[k] - q[k]; return 1 - (e ** 2).sum() / ((q[k] - q[k].mean()) ** 2).sum()
yy, xx = np.mgrid[0:NY, 0:NX]; xc, yc = (xx + 0.5) * G, (yy + 0.5) * G
burner = m & (xc <= 14.6) & (yc >= 9.7) & (yc <= 32)
print("grid R2 whole: mlp %.3f pinn %.3f | burner: mlp %.3f pinn %.3f" % (r2(mlp, C, m), r2(pinn, C, m), r2(mlp, C, burner), r2(pinn, C, burner)))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
ext = (0, NX * G, 0, NY * G)
def panel(ax, A, title, cmap, lo, hi, sub=None):
    im = ax.imshow(A, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear")
    ax.add_patch(Rectangle((0, 9.7), 14.6, 22.3, fill=False, ec="#00b4d8", lw=1.6, ls="--"))
    ax.set_title(title, fontsize=15, fontweight="bold", pad=8)
    if sub: ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top", fontsize=12.5, color="#333")
    ax.set_xlim(0, 33); ax.set_ylim(0, 58.5); ax.set_xticks([0, 15, 30]); ax.set_yticks([0, 20, 40]); ax.tick_params(labelsize=10, colors="#666")
    for s in ax.spines.values(): s.set_visible(False)
    return im
# Figure A: three maps
fig, ax = plt.subplots(1, 3, figsize=(12, 7.2), dpi=200)
panel(ax[0], C, "CFD (original, Fig. 6)", "inferno", 600, 1900, "unseen test image\n(tilt +20°)")
panel(ax[1], mlp, "Data-only MLP", "inferno", 600, 1900, "R² 0.978\nburner zone 0.939")
im = panel(ax[2], pinn, "Hybrid PINN", "inferno", 600, 1900, "R² 0.978\nburner zone 0.942")
cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02); cb.set_label("Temperature (K)")
fig.savefig("F9_tilt20_maps.png", bbox_inches="tight")
# Figure B: error maps
fig, ax = plt.subplots(1, 2, figsize=(8.4, 7.2), dpi=200)
panel(ax[0], np.abs(mlp - C), "|Error| data-only MLP", "magma_r", 0, 250, "MAE 36 K · back pass 13 K")
im = panel(ax[1], np.abs(pinn - C), "|Error| hybrid PINN", "magma_r", 0, 250, "MAE 39 K · back pass 35 K")
cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02); cb.set_label("|Prediction − CFD| (K)")
fig.savefig("F10_tilt20_errors.png", bbox_inches="tight")
print("saved")
