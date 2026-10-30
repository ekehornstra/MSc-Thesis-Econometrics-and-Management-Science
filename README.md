# Conditioning Wind-Precipitation Dependence on Divergence over Lake Victoria: A Copula Approach

Code for the MSc thesis of Eke Hornstra (MSc Econometrics and Management Science, Erasmus University Rotterdam, 2026).
Supervisors: Anastasija Tetereva (Erasmus University Rotterdam) and Louise Nuijens (TU Delft). Second assessor: Jesse Hemerik (Erasmus University Rotterdam).

The thesis asks whether the dependence between wind speed and precipitation over Lake Victoria changes with divergence. It compares parametric and nonparametric copulas as well as conditioning strategies. The thesis also presents two new neural conditional copula models: C-CODINE and C-CopFlow.

## Contents

| Notebook | Content |
|---|---|
| `00_preprocessing` | Reads the raw model output, deseasonalises, fits AR/GARCH models, builds the kernel-conditional pseudo-observations for four blocked folds |
| `01_unconditional_copulas` | Seven parametric families and the unconditional CODINE, per fold |
| `02_binary_conditional_copulas` | The same models fitted separately in a low and a high divergence regime |
| `03_marginal_uniformity_penalty_calibration` | Stage 1 chooses the weight decay of three network groups (C-CODINE, C-CopFlow with affine coupling, C-CopFlow with spline coupling) on ten synthetic designs (fresh draws, penalty off); stage 2 chooses the penalty weight of the same three groups with those weight decays in force. Both stages train only the w-only variant of each group and take the lower median of the best value over the ten designs. The three weight decays and three penalty weights are tuned once and used unchanged for all variants, on every synthetic design and on the real data (`margin_penalty_calibrated.json`, read by notebook 04 and the synthetic notebooks) |
| `04_continuous_conditional_copulas` | Fold x seed batch: parametric, Acar, nonparametric (Gijbels, ECBC) and neural models (C-CODINE, C-CopFlow) with the marginal-uniformity gate |
| `05_comparison_and_mcs` | Cross-validated prediction error (CVPE), Copula Energy Score, Model Confidence Sets and the tail-dependence grid |
| `06_final_figures` | The figures of the thesis that are drawn from the preprocessed data and from the model-comparison results (spin-up check, marginals, diurnal cycle, MCS survivors with the bands of their tau(w) curves, model-implied tail dependence) |
| `07_sensitivity_analysis_spatial_aggregation` | Robustness to the spatial aggregation of the lake |
| `08_copula_families_tail_dependence` | Parametric families at a common Kendall's tau: tail dependence, tail order and joint-extreme curves (needs no data) |
| `AAA01` to `AAA10` | Synthetic study, one notebook per design (five dependence structures, weak and strong) |
| `AAA_summary` | Summary figures and tables over the ten designs, including the combined overview table |
| `RUN_EVERYTHING` | Runs notebooks 00 to 08, then AAA01 to AAA10 and AAA_summary |

The notebooks call the modules next to them:

| Module | Content |
|---|---|
| `preprocessing`, `data_split`, `spatial_aggregation` | Data loading, deseasonalisation, pseudo-observations, blocked folds, spatial aggregation |
| `copula_families`, `conditional_copula`, `acar_glrt` | Parametric families, Acar-type conditional copulas, generalised likelihood-ratio test |
| `codine`, `conditional_copula_flow` | CODINE and the conditional networks C-CODINE and C-CopFlow |
| `fold_pipeline`, `fold_models`, `gate_config` | Fold x seed evaluation, stored models, admissibility gates |
| `model_evaluation`, `model_diagnostics`, `ces_score` | CVPE, bootstrap intervals, Model Confidence Sets, Copula Energy Score, tail diagnostics |
| `synthetic_study`, `synthetic_report` | Synthetic designs, synthetic figures and tables |
| `gijbels_copula`, `ecbc_copula` | Nonparametric conditional estimators: kernel estimator of Gijbels et al. (2011) and the empirical checkerboard Bernstein copula (Lu and Ghosh, 2024) |
| `plotting_core`, `plotting_lake`, `plotting_preprocessing`, `figure_recipes`, `reporting_tables` | Figure style, figures, LaTeX and CSV tables |
| `tau_bands` | Bootstrap bands of the fold-averaged tau(w) of ECBC and Gijbels and the seed range of the networks (notebook 06) |
| `figure_replay`, `replot` | Stores the inputs of every figure and redraws a single figure from them (see below) |
| `run_all`, `run_support` | Headless runner, environment switches, artifact store |

## Requirements

Python with `numpy`, `scipy`, `pandas`, `torch`, `xarray`, `pyvinecopulib`, `arch`, `statsmodels`, `model-confidence-set`, `numba`, `tqdm`, `joblib`, `matplotlib`, `nbconvert` and `ipykernel`. The notebooks are stored with a Jupyter kernel named `jaspy` (the JASMIN environment). On another machine, set `THESIS_KERNEL` to the name of your own kernel (for example `THESIS_KERNEL=python3`) when using `run_all.py`, or change the kernel in Jupyter. Each saved result records the package versions it was produced with (`artifacts/<name>_provenance.json`).

## Data

The data are not included. They are 122 15-day forecasts of a global 5 km numerical weather prediction model coupled to the 1/12-degree NEMO ORCA12 ocean model, initialised every three days from 1 September 2020 to the end of August 2021 (3-hourly 10-metre zonal and meridional wind, hourly precipitation). The output was provided by Eliza Karlowska and Lorenzo Tomassini and is described in the thesis (Section Data). Notebook 00 expects one folder with `lsm.nc` and per-date NetCDF files named `precip_sfc_<date>.nc`, `uwnd_10m_<date>.nc`, `vwnd_10m_<date>.nc` and `st_sfc_<date>.nc`. Point to it with the environment variable `THESIS_RAW_DATA_DIR` (default: the JASMIN folder `/gws/pw/j07/greyzone/users/ekarl20/data/moudp135_sfc_d/std` if it exists, otherwise `./data`; the land-sea mask may be `lsm.nc` or `lsm_sfc_<date>.nc`); `python run_all.py --preflight` checks it.

The synthetic study and notebook 08 need no data.

## Running the code

Put the notebooks and modules in one folder and run from there.

```bash
python run_all.py --preflight        # checks packages and raw data, trains nothing
python run_all.py --everything --fresh  # first complete run, ignores earlier records
python run_all.py --everything          # resumes unchanged successful notebooks; retries failures
python run_all.py --synthetic        # only the synthetic study
python run_all.py --only 04 AAA03    # chosen notebooks by prefix
```

`RUN_EVERYTHING.ipynb` does the same from Jupyter. Each notebook is executed in its own process; the executed copy `<name>_out.ipynb` and a log in `logs/` are written next to the source. Notebook 08 is part of the runner and needs no data. Notebook 03 must run before notebook 04 and the synthetic notebooks, because they read the penalty weights and weight decays it writes; `--everything` takes care of the order. A production run fails if the real fold inputs for notebook 03 or the calibrated penalty file for notebook 04/AAA jobs are missing. `--everything` saves fingerprinted success markers, retries failures and skips only unchanged successful notebooks; it does not start synthetic notebooks after a main-notebook failure. The first complete run should use `--fresh`.

Notebook 04 can be split into stages (`python run_all.py --stage param|acar|nonparam|neural|assemble`); the neural stage can be sharded over several jobs with `THESIS_SHARD`.

| Variable | Effect |
|---|---|
| `THESIS_RAW_DATA_DIR` | Folder with the raw NetCDF files (default: JASMIN folder if present, else `./data`) |
| `THESIS_KERNEL` | Jupyter kernel used by `run_all.py` (default: the kernel stored in each notebook) |
| `THESIS_ARTIFACT_DIR` | Folder for saved results (default `./artifacts`) |
| `THESIS_MAX_WORKERS` | Optional maximum processes per pool, never above CPUs assigned minus two (default: auto) |
| `THESIS_NEURAL_WORKER_GIB` | Estimated peak GiB per neural worker (default 1.5; calibrate from observed RAM) |
| `THESIS_NONPARAM_WORKER_GIB` | Estimated peak GiB per Gijbels/ECBC worker (default 2.5) |
| `THESIS_OTHER_WORKER_GIB` | Estimated peak GiB per other fold/score worker (default 1.5) |
| `THESIS_ACAR_WORKER_GIB` | Estimated peak GiB per Acar CV/bootstrap worker (default 2.0) |
| `THESIS_GLRT_WORKER_GIB` | Estimated peak GiB per GLRT bootstrap worker (default 2.0) |
| `THESIS_SPATIAL_WORKER_GIB` | Estimated peak GiB per spatial-unit worker (default 4.0) |
| `THESIS_SELFCHECK_WORKER_GIB` | Estimated peak GiB per saved-network self-consistency worker (default 2.0) |
| `THESIS_SYNTH_N` | Training sample size of the synthetic study (default 2,000) |
| `THESIS_MARGIN_PENALTY` | One weight of the marginal-uniformity penalty for all three network groups, overriding the calibrated weights; `0` switches the penalty off (default: the weights in `artifacts/margin_penalty_calibrated.json`; missing or outdated production calibration is an error) |
| `THESIS_CAL_LAMBDAS`, `THESIS_CAL_WDS`, `THESIS_CAL_DELTA` | Comma-separated penalty weights and weight decays (grid of stage 2 and stage 1) in notebook 03; optional explicit nonnegative delta, used only when deliberately overriding real-fold matching (recorded in the JSON) |
| `THESIS_ALLOW_UNCALIBRATED_IMPORT` | Set to `1` inside notebooks 01, 02 and 03, which run before the penalty weights exist: `fold_pipeline` then imports with penalty 0 and fallback weight decays instead of raising. Those notebooks train no penalised network; the job fingerprints of the unconditional and regime C-CODINE models depend neither on the penalty weights nor on the calibrated weight decays (they keep weight decay 0), so their results are reused by notebook 04. Do not set it for production runs of notebook 04 |
| `THESIS_SMOKE=1` | Test run with tiny settings and its own folders; never use its results |

**Final-figure/sensitivity cache guards:** notebook 06 fingerprints its Acar curve cache using the actual fold arrays, bandwidths and curve code, so outdated curves are not read after a fit/code change. Notebook 07 now fails if expected spatial cases are missing. The spatial sensitivity CSV is still an intentional resume file; archive or delete it if preprocessing or spatial-analysis assumptions changed before a fresh scientific rerun.

**Parallel execution:** `run_all.py` deliberately executes notebooks sequentially. Independent fits/bootstrap replicates inside each notebook use `resolve_max_workers` (CPU affinity minus two, capped by pending tasks) plus a live cgroup RAM budget from `resolve_memory_workers`. Per-worker GiB values are conservative estimates and should be increased if measured RSS is higher. Notebook 00 intentionally retains its one-worker spin-up and six-worker bootstrap limits due to prior out-of-memory kernel failures. Nested Acar CV inside the parallel synthetic-design fit stays single-process; spatial outer workers likewise run their internal GARCH starts one core at a time. No notebook-level multiprocessing or job arrays are introduced.

The Gijbels and ECBC CVPE uses predictions interpolated across the divergence grid before squaring, with conditional means computed from 32 x 32 masses aggregated from the existing 64 x 64 CES grid. This does not add an extra density grid or training step and remains a different numerical approximation from the other models' Gauss-Legendre CVPE. The Copula Energy Score still uses the shared 64 x 64 grid. The Acar pilot generates AR(1) dependence in the bivariate pairs as well as the covariate; the real-data bootstrap remains iid under its null and is not serial-dependence corrected.

### Redrawing a single figure

Every figure stores the inputs it was drawn from as an artifact `figcall_<figure>` in `artifacts/`. To change how one figure looks, edit its drawing code (`figure_recipes.py`, `plotting_*.py`, `spatial_aggregation.py`, `synthetic_report.py`) and run

```bash
python replot.py --list              # all figures that can be redrawn
python replot.py mcs_survivors       # every figure whose name contains this text
python replot.py --all               # everything
```

The figure is written to the folder `Images <notebook>/` of the run that produced it (`--out FOLDER` writes elsewhere). Nothing is refitted, so a redraw takes seconds. The slow steps (bootstraps, fold-averaged curves, fits) are done before a figure is drawn and are part of the stored inputs.

Results are written to `artifacts/` (saved models and tables of results), `Tables <notebook>/` (every table as a LaTeX file and a CSV file with the same name) and `Images <notebook>/` (figures, PNG only), next to the notebooks. Only figures and tables that appear in the report, or whose numbers are used in the text, are produced; each table is displayed once in its notebook.

## Limitations

The AR and GARCH transformations of notebook 00 are fitted once on the full usable sample and shared by all folds. The held-out observations therefore enter these transformations, while the copula models themselves are trained on the training steps of each fold only.

## Citation

```bibtex
@mastersthesis{hornstra2026divergence,
  author = {Hornstra, Eke},
  title  = {Conditioning Wind-Precipitation Dependence on Divergence over Lake Victoria: A Copula Approach},
  school = {Erasmus University Rotterdam},
  year   = {2026},
  type   = {MSc thesis}
}
```
