# Does the CFD field itself satisfy the physics-loss equations? (25 Nov 2026)

Script: `results/eqcheck.py` (no GPU). Figure: `figures/F8_equation_check.png`.

**Method.** For each of the 7 digitised Fig. 6 images, the temperature field was gridded at 0.1212 m, smoothed with a 1 m Gaussian (the colour bands are 83.5 K steps), and differentiated (∂T/∂y, ∇²T). The two residuals used in the hybrid PINN were evaluated in the upper shaft and back pass, and the unknown coefficients (α, k for the shaft; α, β, T_sink for the back pass) were fitted by non-negative least squares, i.e. the **best** values the equation could take. Interior cells only (edges of the smoothing removed).

| Zone | Fit R² (best physical coefficients) | CFD residual, loss units | Hybrid PINN physics loss after training |
|---|---|---|---|
| Upper shaft | 0.04 | 1.24 | ≤ 0.001 (both zones together) |
| Back pass | 0.38 | 0.41 | ≤ 0.001 |

Medians over the 7 tilts. With unconstrained coefficients the back-pass fit reaches 0.53 but needs a negative diffusivity (α ≈ −3 m²/s) and T_sink ≈ 1,100 K, which are not physical.

**Reading.** The simple energy balance explains almost none of the real temperature variation in the upper shaft and about a third in the back pass. The hybrid PINN drove its physics residual about 1,000× below what the real field has, so it must have bent the temperature field away from the CFD to satisfy an equation the CFD does not obey. This explains why the physics loss made the back pass worse on the unseen +20° test (R² 0.963 → 0.860).

**Caveat.** Derivatives of a band-quantised field depend on the smoothing length (with 2 m smoothing the shaft fit rises to 0.40), but at every smoothing tried the CFD's own residual stays far above what the PINN reached.

**Implication.** A physics loss only helps if the equation describes the data. The next version should use the full energy equation with the real terms (velocity, heat release, radiation) that the 150-case EFB campaign exports per cell.
