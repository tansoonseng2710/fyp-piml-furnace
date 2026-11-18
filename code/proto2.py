"""Autograd stand-in for the Colab notebook's FieldNet + HybridPhysics (same maths), on a subsample, to sanity-check the design."""
import sys, os, time, math
sys.path.insert(0, '../vendor/autograd')
import numpy as np, pandas as pd
import autograd.numpy as anp
from autograd import grad, elementwise_grad as egrad

mode = sys.argv[1]; wpde = float(sys.argv[2]); steps = int(sys.argv[3]); ntr = int(sys.argv[4]); split = sys.argv[5] if len(sys.argv) > 5 else 'split_block'
D = pd.read_csv('furnace_master_dataset.csv.gz'); D = D[~D.is_duplicate_panel].reset_index(drop=True)
XM, YM, TM, TS = 33.0, 58.3, 1100.0, 450.0
rng = np.random.default_rng(0)
tr = D[D[split] == 'train'].sample(ntr, random_state=0); te = D[D[split] == 'test']
fx = lambda d: (d.x_m.values.copy(), d.y_m.values.copy(), d.tilt_deg.values.astype(float), d.T_K.values.copy())
xs, ys, ts, Ts = fx(tr)

NF, SIG = 24, 2.0
B = np.random.default_rng(0).normal(size=(2, NF)) * SIG
LAY = [3 + 2 * NF, 64, 64, 64, 1]
def init(layers, seed=1):
    r = np.random.default_rng(seed); return [(r.normal(0, np.sqrt(1. / a), (a, b)), np.zeros(b)) for a, b in zip(layers[:-1], layers[1:])]
def net(W, x, y, t):
    xn, yn, tn = 2 * x / XM - 1, 2 * y / YM - 1, t / 30.
    p = 2 * math.pi * anp.dot(anp.stack([xn, yn], 1), B)
    h = anp.concatenate([anp.stack([xn, yn, tn], 1), anp.sin(p), anp.cos(p)], 1)
    for w, b in W[:-1]: h = anp.tanh(anp.dot(h, w) + b)
    w, b = W[-1]; return TM + TS * (anp.dot(h, w) + b)[:, 0]

MDOT = 110.07 + 260.54 + 111.182 + 193.5 / 3.6 * (1 - 0.0984)
Cz = lambda A: MDOT * 8.314 / (101325. * 0.0295 * A)
C_S, C_B = Cz(14.63 * 14.63), Cz(14.63 * 12.7); TW = 700.0
pool_s = D[(D.tilt_deg == 0) & (D.zone == 'shaft_upper')][['x_m', 'y_m']].values; pool_b = D[(D.tilt_deg == 0) & (D.zone == 'back_pass')][['x_m', 'y_m']].values

def adam(lossf, P, steps, lr0, log=500):
    g = grad(lossf); flat = lambda P: [a for it in P["W"] for a in it] + ([P["phys"]] if "phys" in P else [])
    F = flat(P); m = [np.zeros_like(p) for p in F]; v = [np.zeros_like(p) for p in F]; t0 = time.time()
    for k in range(1, steps + 1):
        lr = lr0 * 0.5 * (1 + math.cos(math.pi * k / steps)); G = flat(g(P))
        for i, (p, gg) in enumerate(zip(F, G)):
            m[i] = .9 * m[i] + .1 * gg; v[i] = .999 * v[i] + .001 * gg * gg; p -= lr * (m[i] / (1 - .9 ** k)) / (np.sqrt(v[i] / (1 - .999 ** k)) + 1e-8)
        if k % log == 0: print(f'  step {k} loss {float(lossf(P)):.4e} ({time.time()-t0:.0f}s)', flush=True)
    return P

f = lambda W, x, y, t: net(W, x, y, t)
Tx, Ty = egrad(f, 1), egrad(f, 2); Txx, Tyy = egrad(Tx, 1), egrad(Ty, 2)
r2 = np.random.default_rng(3); NC = 600
def pick(pool, n): i = r2.integers(0, len(pool), n); return pool[i, 0], pool[i, 1], r2.uniform(-30, 30, n)
cs = pick(pool_s, NC); cb = pick(pool_b, NC)
P0 = {"W": init(LAY), "phys": np.array([0.0, 0.0, 0.0, 0.0, 0.0])}
if mode == 'pinn_stage2':     # warm start from data-only weights, as in the notebook
    P0["W"] = pickle_W = __import__('pickle').load(open('warm.pkl', 'rb'))
def loss(P, wp=wpde, with_phys=True):
    W = P["W"]; L = anp.mean(((net(W, xs, ys, ts) - Ts) / TS) ** 2)
    if with_phys and wp > 0:
        a_s, k_s, a_b, b_b = [anp.exp(P["phys"][i]) for i in range(4)]; Tsink = 740. + 50. * P["phys"][4]
        x, y, t = cs; T = f(W, x, y, t); lap = Txx(W, x, y, t) + Tyy(W, x, y, t)
        Rs = C_S * T * Ty(W, x, y, t) - a_s * lap + 30. * k_s * ((T / 1000.) ** 4 - (TW / 1000.) ** 4)
        x, y, t = cb; T = f(W, x, y, t); lap = Txx(W, x, y, t) + Tyy(W, x, y, t)
        Rb = -C_B * T * Ty(W, x, y, t) - a_b * lap + b_b * (T - Tsink)
        L = L + wp * (anp.mean((Rs / 100.) ** 2) + anp.mean((Rb / 100.) ** 2))
    return L
def metrics(p, t): e = p - t; return dict(R2=round(float(1 - (e**2).sum() / ((t - t.mean())**2).sum()), 4), MAE=round(float(np.abs(e).mean()), 1), RMSE=round(float(np.sqrt((e**2).mean())), 1))
def report(P, tag):
    p = np.asarray(net(P["W"], *fx(te)[:3])); t = fx(te)[3]; print(tag, 'ALL', metrics(p, t))
    for z in ['shaft_upper', 'back_pass', 'burner_zone', 'hopper', 'horizontal_pass']:
        m = (te.zone == z).values
        if m.any(): print('   ', z, metrics(p[m], t[m]))

if mode == 'mlp':
    P = adam(lambda P: loss(P, with_phys=False), P0, steps, 2e-3); report(P, 'MLP')
    __import__('pickle').dump(P["W"], open('warm.pkl', 'wb'))
elif mode == 'resid0':      # residual magnitude at the data-only solution
    P = {"W": __import__('pickle').load(open('warm.pkl', 'rb')), "phys": np.zeros(5)}
    W = P["W"]
    for name, c, sgn in [('shaft', cs, 1), ('back', cb, -1)]:
        x, y, t = c; T = f(W, x, y, t); ty = Ty(W, x, y, t); lap = Txx(W, x, y, t) + Tyy(W, x, y, t)
        print(name, 'T mean %.0f' % T.mean(), '| adv term rms %.0f K/s' % np.sqrt(np.mean((sgn * (C_S if name == 'shaft' else C_B) * T * ty) ** 2)), '| diffusion(alpha=1) rms %.1f K/s' % np.sqrt(np.mean(lap ** 2)))
else:
    P = adam(loss, P0, steps, 1e-3); report(P, f'PINN w={wpde}')
    print('  learned alpha_s %.3f k_s %.3f alpha_b %.3f beta_b %.3f Tsink %.0f' % (*[float(np.exp(P["phys"][i])) for i in range(4)], 740 + 50 * float(P["phys"][4])))
