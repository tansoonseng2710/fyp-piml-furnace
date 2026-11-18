"""Prototype of the notebook's models using HIPS autograd (stand-in for PyTorch, which can't be installed here)."""
import sys, time, json
import autograd.numpy as anp
import numpy as np
from autograd import grad, elementwise_grad as egrad
import furnace_bench as fb

rng = np.random.default_rng(0)
X, Y, D = fb.make_dataset()
TRAIN_TILTS = [-30, -20, -10, 10, 20, 30]; TEST_TILT = 0
import os
if os.environ.get('EXTRAP'): TRAIN_TILTS = [-30, -20, -10, 0, 10]; TEST_TILT = 30
TM, TS = 1000.0, 400.0

def metrics(p, t):
    e = p - t
    return dict(R2=float(1 - (e**2).sum() / ((t - t.mean())**2).sum()), MAE=float(np.abs(e).mean()),
                MAPE=float(100 * np.abs(e / t).mean()), RMSE=float(np.sqrt((e**2).mean())))

def sample(frac, seed=0):
    r = np.random.default_rng(seed); xs, ys, ts, Ts = [], [], [], []
    for t in TRAIN_TILTS:
        n = X.size; k = max(1, int(frac * n)); ii = r.choice(n, k, replace=False)
        xs.append(X.ravel()[ii]); ys.append(Y.ravel()[ii]); ts.append(np.full(k, float(t))); Ts.append(D[t].ravel()[ii])
    return [np.concatenate(a) for a in (xs, ys, ts, Ts)]

def init(layers, seed=1):
    r = np.random.default_rng(seed)
    return [(r.normal(0, np.sqrt(1.0 / a), (a, b)), np.zeros(b)) for a, b in zip(layers[:-1], layers[1:])]

def net(W, x, y, t):
    h = anp.stack([2 * x - 1, y / 2 - 1, t / 30.0], axis=1)
    for w, b in W[:-1]:
        h = anp.tanh(anp.dot(h, w) + b)
    w, b = W[-1]
    return TM + TS * (anp.dot(h, w) + b)[:, 0]

def adam(lossf, P, steps, lr0, log=500):
    g = grad(lossf); m = [np.zeros_like(p) for p in flat(P)]; v = [np.zeros_like(p) for p in flat(P)]
    t0 = time.time()
    for k in range(1, steps + 1):
        lr = lr0 * (0.1 ** (k / steps))
        G = flat(g(P)); F = flat(P)
        for i, (p, gg) in enumerate(zip(F, G)):
            m[i] = 0.9 * m[i] + 0.1 * gg; v[i] = 0.999 * v[i] + 0.001 * gg * gg
            p -= lr * (m[i] / (1 - 0.9**k)) / (np.sqrt(v[i] / (1 - 0.999**k)) + 1e-8)
        if log and k % log == 0:
            print(f"  step {k} loss {float(lossf(P)):.3e}  ({time.time()-t0:.0f}s)", flush=True)
    return P

def flat(P):
    out = []
    for item in P["W"]:
        out += [item[0], item[1]]
    if "phys" in P: out.append(P["phys"])
    return out

def predict(P, t):
    return np.asarray(net(P["W"], X.ravel(), Y.ravel(), np.full(X.size, float(t)))).reshape(X.shape)

LAYERS = [3, 48, 48, 48, 1]

def train_data_only(data, steps, extra=None, w_extra=1.0):
    xs, ys, ts, Ts = data
    P = {"W": init(LAYERS)}
    def loss(P):
        L = anp.mean(((net(P["W"], xs, ys, ts) - Ts) / TS) ** 2)
        if extra is not None:
            ex, ey, et, eT = extra
            L = L + w_extra * anp.mean(((net(P["W"], ex, ey, et) - eT) / TS) ** 2)
        return L
    return adam(loss, P, steps, 3e-3)

def train_pinn(data, steps, extra=None, w_extra=1.0, w_pde=1.0, w_bc=1.0, ncol=1500, seed=3, W0=None):
    xs, ys, ts, Ts = data
    r = np.random.default_rng(seed)
    P = {"W": W0 if W0 is not None else init(LAYERS), "phys": np.array([np.log(0.02), np.log(0.15)])}   # alpha, beta start 2x off
    f = lambda W, x, y, t: net(W, x, y, t)
    Tx, Ty = egrad(f, 1), egrad(f, 2)
    Txx, Tyy = egrad(Tx, 1), egrad(Ty, 2)
    # fixed collocation set in source-free regions (below / above the flame band), all tilts incl. unseen ones
    def col(n):
        x = r.uniform(0, 1, n); t = r.uniform(-30, 30, n)
        lo = r.uniform(0, fb.FLAME_BAND[0], n // 3); hi = r.uniform(fb.FLAME_BAND[1], fb.H, n - n // 3)
        return x, np.concatenate([lo, hi]), t
    cx, cy, ct = col(ncol)
    cu = fb.uy(cx, ct)
    nb = 300; bt = r.uniform(-30, 30, nb)
    wall_y = r.uniform(0, fb.H, nb); wall_x = np.where(r.random(nb) < 0.5, 0.0, 1.0)
    bot_x = r.uniform(0, 1, nb); top_x = r.uniform(0, 1, nb)
    def loss(P):
        W = P["W"]; a, b = anp.exp(P["phys"][0]), anp.exp(P["phys"][1])
        L = anp.mean(((net(W, xs, ys, ts) - Ts) / TS) ** 2)
        if extra is not None:
            ex, ey, et, eT = extra
            L = L + w_extra * anp.mean(((net(W, ex, ey, et) - eT) / TS) ** 2)
        T = f(W, cx, cy, ct)
        res = cu * Ty(W, cx, cy, ct) - a * (Txx(W, cx, cy, ct) + Tyy(W, cx, cy, ct)) + b * (T - fb.T_WALL)
        L = L + w_pde * anp.mean((res / TS) ** 2)
        Lbc = anp.mean(((net(W, wall_x, wall_y, bt) - fb.T_WALL) / TS) ** 2) \
            + anp.mean(((net(W, bot_x, np.zeros(nb), bt) - fb.T_IN) / TS) ** 2) \
            + anp.mean((Ty(W, top_x, np.full(nb, fb.H), bt) / TS) ** 2)
        return L + w_bc * Lbc
    P = adam(loss, P, steps, 3e-3)
    return P

def sensors(t=TEST_TILT, noise=5.0, seed=7):
    pts = [(0.12, 1.8), (0.88, 1.8), (0.12, 2.6), (0.88, 2.6), (0.3, 3.4), (0.7, 3.4), (0.5, 3.9), (0.5, 0.4)]
    r = np.random.default_rng(seed)
    xi = [int(round(px * (fb.NX - 1))) for px, _ in pts]; yi = [int(round(py / fb.H * (fb.NY - 1))) for _, py in pts]
    T = np.array([D[t][i, j] for i, j in zip(xi, yi)]) + r.normal(0, noise, len(pts))
    return np.array([X[i, j] for i, j in zip(xi, yi)]), np.array([Y[i, j] for i, j in zip(xi, yi)]), np.full(len(pts), float(t)), T

if __name__ == "__main__":
    which = sys.argv[1]; steps = int(sys.argv[2]); frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    data = sample(frac)
    truth = D[TEST_TILT]
    t0 = time.time()
    if which == "lin":
        print("lin", metrics(0.5 * (D[-10] + D[10]), truth))
    elif which == "mlp":
        P = train_data_only(data, steps); print("MLP", frac, metrics(predict(P, TEST_TILT), truth), f"{time.time()-t0:.0f}s")
    elif which == "pinn":
        warm = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        wp = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        W0 = train_data_only(data, warm)["W"] if warm else None
        P = train_pinn(data, steps, W0=W0, w_pde=wp); print("PINN", frac, metrics(predict(P, TEST_TILT), truth), "alpha,beta", np.exp(P["phys"]), f"{time.time()-t0:.0f}s")
    elif which == "mlp_s":
        P = train_data_only(data, steps, extra=sensors(), w_extra=float(os.environ.get("WS","2.0"))); print("MLP+sensors", frac, metrics(predict(P, TEST_TILT), truth))
    elif which == "pinn_s":
        S = sensors(); W0 = train_data_only(data, 2000, extra=S, w_extra=float(os.environ.get("WS","2.0")))["W"]
        P = train_pinn(data, steps, extra=S, w_extra=float(os.environ.get("WS","2.0")), W0=W0, w_pde=0.1); print("PINN+sensors", frac, metrics(predict(P, TEST_TILT), truth), np.exp(P["phys"]))

# ---------- paper-style "sandwich" MLP (Xue et al. 2024): inputs include neighbour-tilt temperatures ----------
from itertools import combinations
def sandwich_data(frac, seed=0):
    r = np.random.default_rng(seed); F, Tt = [], []
    for a, m, b in combinations(TRAIN_TILTS, 3):     # sorted triplets, middle = target
        k = int(frac * X.size); ii = r.choice(X.size, k, replace=False)
        F.append(np.stack([X.ravel()[ii], Y.ravel()[ii], np.full(k, m), np.full(k, a), np.full(k, b), D[a].ravel()[ii], D[b].ravel()[ii]], 1)); Tt.append(D[m].ravel()[ii])
    return np.concatenate(F), np.concatenate(Tt)
def snet(W, F):
    h = anp.stack([2*F[:,0]-1, F[:,1]/2-1, F[:,2]/30, F[:,3]/30, F[:,4]/30, (F[:,5]-TM)/TS, (F[:,6]-TM)/TS], 1)
    for w, b in W[:-1]: h = anp.tanh(anp.dot(h, w) + b)
    w, b = W[-1]; return TM + TS * (anp.dot(h, w) + b)[:, 0]
def train_sandwich(frac, steps):
    F, Tt = sandwich_data(frac); P = {"W": init([7, 48, 48, 48, 1])}
    return adam(lambda P: anp.mean(((snet(P["W"], F) - Tt) / TS) ** 2), P, steps, 3e-3)
if __name__ == "__main__" and which == "sandwich":
    P = train_sandwich(frac, steps)
    Ft = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size), np.full(X.size, -10.), np.full(X.size, 10.), D[-10].ravel(), D[10].ravel()], 1)
    print("Sandwich MLP", frac, metrics(np.asarray(snet(P["W"], Ft)).reshape(X.shape), truth))
