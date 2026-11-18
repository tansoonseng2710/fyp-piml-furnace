import numpy as np, pandas as pd, json
import digitise as dg

S = dg.H_BOILER / 481.0                 # metres per pixel (panel 1 spans 481 px for 58.3 m)
im = dg.load(); cols, lo, hi, mid = dg.band_colours(im); panels = dg.find_panels(im)

rows = []; bands = {}
for pid, (sl, comp) in enumerate(panels, 1):
    band, dist, m, ok, (r0, r1, c0, c1) = dg.digitise_panel(im, sl, comp, cols)
    bands[pid] = (band, ok, r0, c0)
    rr, cc = np.nonzero(ok)
    rows.append(pd.DataFrame(dict(
        panel=pid, tilt_deg=dg.PANELS[pid], is_duplicate_panel=(pid == 5),
        px_row=rr, px_col=cc, x_m=cc * S, y_m=dg.H_BOILER - rr * S,
        band=band[rr, cc], T_lo_K=lo[band[rr, cc]], T_hi_K=hi[band[rr, cc]], T_K=mid[band[rr, cc]],
        colour_dist=dist[rr, cc].round(1))))
df = pd.concat(rows, ignore_index=True)

# ---------- QA: the two 0-degree panels (1 and 5) are the same CFD case drawn twice ----------
b1, ok1, r1_, c1_ = bands[1]; b5, ok5, r5_, c5_ = bands[5]
h = min(b1.shape[0], b5.shape[0]); w = min(b1.shape[1], b5.shape[1])
both = ok1[:h, :w] & ok5[:h, :w]
agree = (b1[:h, :w] == b5[:h, :w])[both]
diff = np.abs(b1[:h, :w].astype(int) - b5[:h, :w].astype(int))[both]
qa = dict(n_pixels_compared=int(both.sum()), exact_band_agreement_pct=float(100 * agree.mean()),
          within_1_band_pct=float(100 * (diff <= 1).mean()), mean_abs_diff_K=float(diff.mean() * 83.5))
print('QA 0deg duplicate:', qa)

# ---------- zones (approximate, from geometry; used for per-zone reporting and for the physics term) ----------
FLAME_TOP = 32.0          # m. Above this the burner heat release is assumed over (see notebook).
x, y = df.x_m.values, df.y_m.values
zone = np.full(len(df), 'burner_zone', dtype=object)
zone[(y < 9.6)] = 'hopper'
zone[(x <= 14.8) & (y >= FLAME_TOP) & (y < 43.0)] = 'shaft_upper'
zone[(x <= 14.8) & (y >= 43.0) & (y < 47.5)] = 'nose'
zone[(x > 14.8) & (y < 47.5)] = 'back_pass'
zone[(y >= 47.5)] = 'horizontal_pass'
df['zone'] = zone

# ---------- splits (exactly 9/10 train, 1/10 test) ----------
train = df[~df.is_duplicate_panel].copy()
rng = np.random.default_rng(2026)
sp_rand = pd.Series('train', index=df.index)
for t, g in train.groupby('tilt_deg'):                    # random 10% of points within every tilt
    idx = rng.choice(g.index.values, int(round(0.10 * len(g))), replace=False)
    sp_rand.loc[idx] = 'test'
sp_blk = pd.Series('train', index=df.index)
BS = 4.0
bx, by = (df.x_m // BS).astype(int), (df.y_m // BS).astype(int)         # 4 m x 4 m spatial blocks, identical for every tilt
key = bx.astype(str) + '_' + by.astype(str)
use = ~df.is_duplicate_panel; total = int(use.sum())
cnt_blk = key[use].value_counts()
dom = df[use].groupby(key[use]).zone.agg(lambda z: z.value_counts().index[0])      # dominant zone of each block
ZONES = ['shaft_upper', 'back_pass', 'burner_zone', 'hopper', 'horizontal_pass']
best = None
for seed in range(4000):
    r = np.random.default_rng(seed); chosen = []; cnt = 0
    for z in ZONES:                                                      # at least one full block in every zone
        cand = [b for b in cnt_blk.index if dom[b] == z and cnt_blk[b] >= 0.85 * 7 * 1089 / 7]
        b = cand[r.integers(len(cand))]; chosen.append(b); cnt += cnt_blk[b]
    rest = [b for b in r.permutation(cnt_blk.index.values) if b not in chosen and cnt_blk[b] >= 900]
    for b in rest:
        if cnt >= 0.10 * total: break
        chosen.append(b); cnt += cnt_blk[b]
    gap = abs(cnt / total - 0.10)
    if best is None or gap < best[0]: best = (gap, chosen, seed)
chosen = best[1]; print('block seed', best[2], 'blocks', chosen, 'zones', [dom[b] for b in chosen])
sp_blk[key.isin(chosen) & use] = 'test'
df['split_random'] = np.where(df.is_duplicate_panel, 'excluded_duplicate', sp_rand)
df['split_block'] = np.where(df.is_duplicate_panel, 'excluded_duplicate', sp_blk)
for c in ['split_random', 'split_block']:
    print(c, df[c].value_counts().to_dict(), ' test share of usable: %.2f%%' % (100 * (df[c] == 'test').sum() / total))
print(df[~df.is_duplicate_panel].groupby('tilt_deg').split_block.apply(lambda s: round((s == 'test').mean() * 100, 2)).to_dict())

df.to_csv('furnace_master_dataset.csv.gz', index=False, float_format='%.4f', compression='gzip')
json.dump(dict(qa=qa, scale_m_per_px=S, n_rows=len(df), n_usable=total, blocks=chosen, flame_top_m=FLAME_TOP, block_size_m=BS), open('build_info.json', 'w'), indent=1)
print(df.shape); print(df.head()); print(df.groupby('zone').size()); print(df[~df.is_duplicate_panel].groupby('tilt_deg').size())
