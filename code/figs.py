import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
RED, INK, MUTED, GRID, LIGHT = '#9B1C1C', '#1F2328', '#6B7280', '#E5E7EB', '#F3E8E8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.edgecolor': GRID, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
D = pd.read_csv('../master/out/furnace_master_dataset.csv.gz'); D = D[~D.is_duplicate_panel]

# F1: seven digitised fields
fig, ax = plt.subplots(1, 7, figsize=(16, 6.2), sharey=True)
for a, t in zip(ax, [-30, -20, -10, 0, 10, 20, 30]):
    g = D[D.tilt_deg == t]
    sc = a.scatter(g.x_m, g.y_m, c=g.T_K, s=0.35, cmap='inferno', vmin=600, vmax=1900, rasterized=True)
    a.set_title(f'{t:+d}°', color=INK, fontsize=15); a.set_aspect('equal'); a.set_xticks([0, 15, 30]); a.tick_params(labelsize=9)
    for s in a.spines.values(): s.set_visible(False)
ax[0].set_ylabel('height (m)')
cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01); cb.set_label('Temperature (K)'); cb.outline.set_visible(False)
fig.savefig('F1_digitised_fields.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close()

# F2: zones and held-out blocks at 0 deg
g = D[D.tilt_deg == 0]
fig, ax = plt.subplots(1, 2, figsize=(10, 7))
zc = {'hopper': '#9CA3AF', 'burner_zone': '#7F1D1D', 'shaft_upper': '#DC6B6B', 'nose': '#D1D5DB', 'back_pass': '#E8B4B4', 'horizontal_pass': '#4B5563'}
names = {'hopper': 'Hopper', 'burner_zone': 'Burner zone', 'shaft_upper': 'Upper shaft (physics)', 'nose': 'Nose', 'back_pass': 'Back pass (physics)', 'horizontal_pass': 'Horizontal pass'}
for z, c in zc.items():
    q = g[g.zone == z]; ax[0].scatter(q.x_m, q.y_m, s=0.4, c=c, label=names[z], rasterized=True)
ax[0].legend(markerscale=14, fontsize=10, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2); ax[0].set_title('Furnace zones', color=INK)
ax[1].scatter(g.x_m, g.y_m, s=0.4, c='#E5E7EB', rasterized=True); te = g[g.split_block == 'test']
ax[1].scatter(te.x_m, te.y_m, s=0.4, c=RED, rasterized=True); ax[1].set_title('Held-out 4 m blocks (1/10 test)', color=INK)
for a in ax: a.set_aspect('equal'); a.set_xlabel('x (m)')
ax[0].set_ylabel('height (m)')
fig.savefig('F2_zones_split.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close()

# F3: hybrid PINN architecture
fig, ax = plt.subplots(figsize=(15, 6.4)); ax.set_xlim(0, 15); ax.set_ylim(0, 6.4); ax.axis('off')
def box(x, y, w, h, title, sub, fc='white', ec=RED, tc=INK, fs=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=2))
    ax.text(x + w/2, y + h*0.68, title, ha='center', va='center', fontsize=fs, weight='bold', color=tc)
    ax.text(x + w/2, y + h*0.30, sub, ha='center', va='center', fontsize=10.5, color=MUTED if tc == INK else tc)
def arrow(x1, y1, x2, y2, c=INK, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=c, lw=1.8, ls=ls))
box(0.2, 3.7, 2.2, 1.6, 'Inputs', 'x, y (m)\nburner tilt (°)')
box(2.9, 3.7, 2.7, 1.6, 'Fourier features', '48 sin/cos of (x, y)\nfor sharp flame edges', fs=13)
box(6.1, 3.7, 2.6, 1.6, 'FieldNet', '5 hidden layers × 128\ntanh activation')
box(9.2, 3.7, 2.0, 1.6, 'Output', 'Temperature T (K)')
for x1, x2 in [(2.4, 2.9), (5.6, 6.1), (8.7, 9.2)]: arrow(x1, 4.5, x2, 4.5)
box(11.8, 4.55, 3.0, 1.3, 'Data loss', 'MSE vs digitised CFD\n(9/10 train points)', fc=LIGHT, fs=13)
box(11.8, 2.85, 3.0, 1.3, 'Physics loss × 0.1', 'energy-balance residual\nafter 1,500 warm-up steps', fc=RED, ec=RED, tc='white', fs=13)
arrow(11.2, 4.75, 11.8, 5.2); arrow(11.2, 4.25, 11.8, 3.5, RED)
ax.add_patch(FancyBboxPatch((0.2, 0.2), 11.0, 2.9, boxstyle='round,pad=0.02,rounding_size=0.12', fc='#FAFAFA', ec=GRID, lw=1.5))
ax.text(0.45, 2.7, 'Physics residual (autograd derivatives of T), applied only outside the burner zone', fontsize=12, weight='bold', color=INK)
ax.text(0.45, 2.05, 'Upper shaft (gas up):   C·T·∂T/∂y − α∇²T + 30k·[(T/1000)⁴ − 0.7⁴] = 0', fontsize=11.5, color=INK, family='DejaVu Sans Mono')
ax.text(0.45, 1.5, 'Back pass (gas down):  −C·T·∂T/∂y − α∇²T + β·(T − T_sink) = 0', fontsize=11.5, color=INK, family='DejaVu Sans Mono')
ax.text(0.45, 0.85, 'C from Table 2 mass flow (530 kg/s → about 10 m/s).  α, k, β, T_sink are learned.', fontsize=10.5, color=MUTED)
ax.text(0.45, 0.45, 'Collocation points sit on the real geometry at random tilts between −30° and +30°.', fontsize=10.5, color=MUTED)
ax.plot([13.3, 13.3, 11.25], [2.85, 1.65, 1.65], color=RED, lw=1.8)
fig.savefig('F3_hybrid_pinn.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close()

# F4a: synthetic benchmark, unseen 0 deg tilt, MAPE
def bars(labels, vals, hi, fname, xlabel, fmt, title, floor=None, floor_lbl=None, xmax=None):
    fig, ax = plt.subplots(figsize=(8.5, 0.75 * len(labels) + 1.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.6, color=[RED if i in hi else '#C9CDD3' for i in range(len(labels))])
    for yi, v in zip(y, vals): ax.text(v + (xmax or max(vals)) * 0.015, yi, fmt.format(v), va='center', color=INK, fontsize=12)
    ax.set_yticks(y); ax.set_yticklabels(labels, color=INK, fontsize=12); ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    if floor: ax.axvline(floor, color=INK, ls='--', lw=1.2); ax.set_xlabel(xlabel + '   ·   dashed line: ' + floor_lbl)
    if xmax: ax.set_xlim(0, xmax)
    ax.set_title(title, loc='left', color=INK, fontsize=13)
    fig.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
bars(['Linear interpolation', 'Paper-style MLP', 'Data-only MLP', 'Hybrid PINN'], [2.57, 2.28, 1.37, 1.25], {3},
     'F4_synthetic_results.png', 'MAPE at unseen 0° tilt (%)  ·  lower is better', '{:.2f}%', 'Synthetic furnace benchmark (5% of points)', xmax=3.0)
bars(['Data-only MLP (0.5% data)', 'Hybrid PINN (0.5% data)'], [1.49, 1.07], {1}, 'F5_sparse_results.png',
     'MAPE at unseen 0° tilt (%)', '{:.2f}%', 'With very sparse data, physics helps most', xmax=1.8)
bars(['kNN, random 9/10 split', 'kNN, block 9/10 split', 'kNN (x, y, tilt), unseen 0° tilt'], [8.2, 21.5, 26.3], set(),
     'F6_knn_baselines.png', 'MAE on test set (K)', '{:.1f} K', 'Digitised paper data: the baseline to beat', floor=20.9, floor_lbl='≈21 K quantisation floor', xmax=32)
bars(['Data-only MLP', 'Hybrid PINN (w = 0.1)'], [43.4, 21.6], {1}, 'F7_backpass_proto.png',
     'MAE in held-out back-pass block (K)', '{:.1f} K', 'Small CPU check (20k points, one seed)', xmax=52)
print('ok')
