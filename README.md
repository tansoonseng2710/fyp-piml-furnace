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
| `PIML_Furnace_Master.ipynb` | Main notebook: kNN and data-only MLP baselines, hybrid PINN, trained on 9/10 and tested on 1/10 (random, block and unseen-tilt splits) |
| `data/furnace_master_dataset.csv.gz` | Fig. 6 of the anchor paper digitised: 506,706 points (x, y, burner tilt, temperature) for 7 tilts, with split columns |
| `data/furnace_master.xlsx` | All numbers from the paper's Tables 1–6, data dictionary, digitisation check, split summary |
| `notebooks/PIML_Furnace_Starter.ipynb` | First notebook: the same method on a synthetic 2D furnace with known physics |
| `code/` | Scripts that built the dataset, workbook, notebooks and figures |
| `figures/` | Figures used in the progress presentation |

## Limits of the data

- Digitised from a colour contour plot with 20 bands, 83.5 K wide, so temperatures are rounded (error floor about 21 K).
- One vertical plane (2D), not the full 3D furnace. The paper's raw CFD data is only available on request.
- The temperature field values were read from the published figure for research use; credit belongs to the original authors.
