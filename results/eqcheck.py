import sys; POS = len(sys.argv) > 1
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
D = pd.read_csv("../master/furnace_master_dataset.csv.gz"); D = D[~D.is_duplicate_panel.astype(bool)]
dx = 0.1212
MDOT = 110.07 + 260.54 + 111.182 + 193.5 / 3.6 * (1 - 0.0984)
C = lambda A: MDOT * 8.314 / (101325.0 * 0.0295 * A)
CS, CB = C(14.63 * 14.63), C(14.63 * 12.7)
out = []
for sig_m in [1.0]:
  for tilt, g in D.groupby("tilt_deg"):
    r = g.px_row.values - g.px_row.min(); c = g.px_col.values - g.px_col.min()
    H, W = r.max() + 1, c.max() + 1
    T = np.zeros((H, W)); M = np.zeros((H, W)); Z = np.full((H, W), "", object)
    T[r, c] = g.T_K.values; M[r, c] = 1; Z[r, c] = g.zone.values
    s = sig_m / dx
    Ts = gaussian_filter(T * M, s) / np.maximum(gaussian_filter(M, s), 1e-9)
    # rows increase downward -> y up = -row direction
    Ty = -np.gradient(Ts, dx, axis=0); Tx = np.gradient(Ts, dx, axis=1)
    lap = np.gradient(np.gradient(Ts, dx, axis=0), dx, axis=0) + np.gradient(np.gradient(Ts, dx, axis=1), dx, axis=1)
    # keep interior cells (mask eroded by 3 sigma so smoothing edges are excluded)
    from scipy.ndimage import binary_erosion
    inner = binary_erosion(M > 0, iterations=int(3 * s) + 1)
    for zone in ["shaft_upper", "back_pass"]:
        k = inner & (Z == zone)
        if k.sum() < 50: continue
        t, ty, lp = Ts[k], Ty[k], lap[k]
        if zone == "shaft_upper":
            conv = CS * t * ty                                   # = alpha*lap - 30k*((T/1000)^4 - 0.7^4)
            A = np.c_[lp, -30 * ((t / 1000) ** 4 - 0.7 ** 4)]
        else:
            conv = -CB * t * ty                                  # = alpha*lap - beta*T + beta*Tsink
            A = np.c_[lp, -t, np.ones_like(t)]
        from scipy.optimize import nnls
        coef, _ = nnls(A, conv) if POS else (np.linalg.lstsq(A, conv, rcond=None)[0], 0)
        res = conv - A @ coef
        r2 = 1 - (res ** 2).sum() / ((conv - conv.mean()) ** 2).sum()
        out.append(dict(sigma_m=sig_m, tilt=tilt, zone=zone, n=int(k.sum()), conv_rms=float(np.sqrt((conv ** 2).mean())),
                        res_rms=float(np.sqrt((res ** 2).mean())), fit_R2=float(r2), coef=np.round(coef, 3).tolist(),
                        res_loss_units=float(((res / 100) ** 2).mean())))
R = pd.DataFrame(out)
pd.set_option("display.width", 200)
print(R.groupby(["sigma_m", "zone"])[["conv_rms", "res_rms", "fit_R2", "res_loss_units"]].median().round(4))
print(R[R.sigma_m == 1.0][["tilt", "zone", "n", "conv_rms", "res_rms", "fit_R2", "coef", "res_loss_units"]].round(4).to_string())
R.to_csv("eqcheck_pos.csv" if POS else "eqcheck_free.csv", index=False)
