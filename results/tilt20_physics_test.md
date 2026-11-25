# Does the physics loss help? Unseen +20° tilt test (25 Nov 2026)

Run on Google Colab, Tesla T4, `PIML_Furnace_Master.ipynb` (commit d168ea4).

**Setup:** one network (FieldNet: Fourier features + 5 × 128 tanh), trained on 6 tilt images (−30, −20, −10, 0, +10, +30°), tested on the unseen +20° image. Data-only MLP = MSE loss; hybrid PINN = MSE + 0.1 × energy-balance loss (upper shaft and back pass only). Same data, steps (6,000), learning rate and seeds; 3 seeds each. Mean ± standard deviation over seeds.

## Results on the +20° image

| Region | Model | R² | MAE (K) | MAPE (%) | RMSE (K) |
|---|---|---|---|---|---|
| Whole image | kNN reference | 0.9625 | 39.5 | 3.2 | 70.7 |
| Whole image | MLP (data only) | 0.9783 ± 0.0022 | 36.4 ± 1.8 | 2.9 | 53.8 |
| Whole image | Hybrid PINN | 0.9779 ± 0.0009 | 38.9 ± 0.4 | 3.2 | 54.3 |
| Burner zone (focus) | kNN reference | 0.8522 | 59.8 | 4.1 | 85.5 |
| Burner zone (focus) | MLP (data only) | 0.9386 ± 0.0037 | 42.3 ± 0.9 | 2.7 | 55.1 |
| Burner zone (focus) | Hybrid PINN | 0.9420 ± 0.0060 | 41.5 ± 2.2 | 2.6 | 53.5 |
| Upper shaft (physics) | MLP (data only) | 0.7786 ± 0.0505 | 38.6 ± 5.8 | 2.4 | 48.8 |
| Upper shaft (physics) | Hybrid PINN | 0.7368 ± 0.0096 | 42.6 ± 1.1 | 2.6 | 53.5 |
| Back pass (physics) | MLP (data only) | 0.9630 ± 0.0002 | 13.3 ± 0.2 | 1.6 | 21.3 |
| Back pass (physics) | Hybrid PINN | 0.8602 ± 0.0013 | 34.7 ± 0.5 | 4.4 | 41.4 |

## Hybrid minus data-only

| Region | R² gain | MAE change | Hybrid better in |
|---|---|---|---|
| Whole image | −0.0004 | 7.1% worse | 2 of 3 seeds |
| Burner zone | +0.0035 | 1.8% better | 3 of 3 seeds |
| Upper shaft | −0.0418 | 11.7% worse | 1 of 3 seeds |
| Back pass | −0.1028 | 160% worse | 0 of 3 seeds |

15 fixed points in the burner zone: hybrid closer at 8 of 15; mean error 48 K (data-only) vs 49 K (hybrid).

Learned coefficients (3 seeds): alpha_shaft 0.39–0.50, k_rad 0.37–0.40, alpha_back 0.76–0.89, beta_back 1.63–1.72, T_sink ≈ 713 K.

## Reading

- Both neural models beat the kNN reference (R² 0.978 vs 0.963 overall, 0.94 vs 0.85 in the burner zone).
- On the unseen tilt, the physics loss in its current form does not improve the overall result. The small burner-zone gain (+0.0035 R²) is smaller than the seed-to-seed spread, so it is not a reliable improvement.
- In the regions where the physics is applied, it makes the prediction worse, most of all in the back pass. The equations' assumptions (vertical flow with speed from total mass flow, uniform tube heat sink, textbook radiation) do not match this CFD field closely enough; the network learned k_rad ≈ 0.4, well below the textbook 1.
- Labels are colour bands 83.5 K wide (MAE floor ≈ 21 K). Not comparable with the paper's Table 4 (3-D CFD, 18 inputs).
