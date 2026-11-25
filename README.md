# Hybrid PINN for boiler furnace temperature fields

Final year project, KIG 4054, Universiti Malaya (2026/2027).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tansoonseng2710/fyp-piml-furnace/blob/main/PIML_Furnace_Master.ipynb)

**Goal:** rebuild the temperature field inside a coal-fired boiler furnace from CFD data with a physics-informed neural network (hybrid PINN): a network trained on data plus an energy-balance loss.

**Anchor paper:** Xue et al. (2024), *Energy* 286, 129568, https://doi.org/10.1016/j.energy.2023.129568

## Run it

Click **Open in Colab**, then *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The notebook downloads the dataset from this repo by itself.

## What is here

| Path | What it is |
|---|---|
| `PIML_Furnace_Master.ipynb` | Main notebook: does the physics loss help? Same network trained with and without the physics loss on 6 tilt images, tested on the unseen +20° image (3 seeds). Reports R², MAE, MAPE, RMSE overall and in the burner zone, plus point-by-point errors |
| `notebooks/PIML_Furnace_Master_v1_splits.ipynb` | Earlier version: random, block and unseen-0° splits |
| `data/furnace_master_dataset.csv.gz` | Fig. 6 of the anchor paper digitised: 506,706 points (x, y, burner tilt, temperature) for 7 tilts, with split columns |
| `data/furnace_master.xlsx` | All numbers from the paper's Tables 1–6, data dictionary, digitisation check, split summary |
| `notebooks/PIML_Furnace_Starter.ipynb` | First notebook: the same method on a synthetic 2D furnace with known physics |
| `code/` | Scripts that built the dataset, workbook, notebooks and figures |
| `figures/` | Figures used in the progress presentation |

## Limits of the data

- 506,706 rows are pixels (0.12 m) of 7 CFD pictures, not 506,706 separate CFD results. On a 1 m grid this is about 8,000 points.
- Digitised from a colour contour plot with 20 bands, 83.5 K wide, so temperatures are rounded (error floor about 21 K).
- One vertical plane (2D), not the full 3D furnace. The paper's raw CFD data is only available on request.
- The temperature field values were read from the published figure for research use; credit belongs to the original authors.
