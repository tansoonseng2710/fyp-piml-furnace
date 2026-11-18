"""Digitise Fig. 6 of Xue et al. (2024), Energy 286, 129568: CFD temperature on the furnace central vertical plane, 7 burner tilts.
Pixel colour -> nearest of the 20 colour-bar bands (330-2000 K, 83.5 K per band) -> band-midpoint temperature."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

IMG = '../paper/img/im-008.png'
EDGES = np.array([2000,1917,1833,1750,1666,1583,1499,1416,1332,1249,1165,1082,998,915,831,748,664,581,497,414,330], float)
BAR_Y0, BAR_Y1 = 37, 1057
H_BOILER, W_BOILER = 58.3, 14.63            # m, from Section 3.1 of the paper
PANELS = {  # label -> tilt (deg); order matches detected panels (row-major)
    1: 0, 2: 10, 3: 20, 4: 30, 5: 0, 6: -10, 7: -20, 8: -30}

def load():
    return np.asarray(Image.open(IMG).convert('RGB')).astype(int)

def band_colours(im):
    h = (BAR_Y1 - BAR_Y0) / 20
    cols = np.array([im[int(BAR_Y0 + h * (k + .5)), 38:48].mean(0) for k in range(20)])
    lo = EDGES[1:]; hi = EDGES[:-1]
    return cols, lo, hi, (lo + hi) / 2

def find_panels(im):
    sat = im.max(2) - im.min(2); mask = sat > 40; mask[:, :100] = False
    lab, n = ndi.label(mask); sl = ndi.find_objects(lab)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    P = [(s, lab == i + 1) for i, (s, a) in enumerate(zip(sl, sizes)) if a > 5000]
    P.sort(key=lambda p: (round(p[0][0].start / 300), p[0][1].start))
    return P

def digitise_panel(im, sl, comp, cols, maxdist=40.0):
    r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
    sub = im[r0:r1, c0:c1]; m = comp[r0:r1, c0:c1]
    d = np.sqrt(((sub[:, :, None, :] - cols[None, None, :, :]) ** 2).sum(-1))
    band = d.argmin(2); dist = d.min(2)
    ok = m & (dist <= maxdist)
    return band, dist, m, ok, (r0, r1, c0, c1)

if __name__ == '__main__':
    im = load(); cols, lo, hi, mid = band_colours(im); P = find_panels(im)
    print(len(P), 'panels')
    for i, (sl, comp) in enumerate(P, 1):
        band, dist, m, ok, bb = digitise_panel(im, sl, comp, cols)
        print(i, PANELS[i], 'bbox', bb, 'h px', bb[1]-bb[0], 'w px', bb[3]-bb[2], 'mask px', int(m.sum()), 'kept', int(ok.sum()), f'({100*ok.sum()/m.sum():.1f}%)',
              'median dist', np.median(dist[m]).round(1), 'p95', np.percentile(dist[m], 95).round(1))
