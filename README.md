# Neural SERS decomposition and cross-window consensus selection

Code and processed data for **Beyond Reconstruction Loss: Cross-Window Consensus Selection for Neural SERS Analyte-Spectrum Recovery**.

The method fits an analyte-spectrum branch and a coefficient branch to ordered mixture spectra with a supplied fixed background. Candidate configurations are ranked by analyte recovery within each validation window. The first configuration to enter the intersection of retained candidates across all validation windows is carried into held-out testing.

## What is included

- The final 7,560-configuration grid and exact processed inputs for 37 validation and 33 held-out test windows.
- All 279,720 validation recovery scores and all 529,200 validation/test reconstruction residuals, including the identifiers needed to reproduce selection.
- The final neural model, training loss, optimizer, reconstruction-loss comparator, fixed-background MCR-ALS, and prior dual-network comparator implementations.
- Saved results and replay support for the published three-seed robustness, hidden-size, window-size, and representative ablation analyses.
- Numerical inputs for the main and supporting figures, table-generation code, and the approved figure images.
- A runnable synthetic example and CSV interface for fitting your own processed window.

This is a final-result release. Development experiments, exploratory seed-selection workflows, obsolete grids, cluster administration scripts, and manuscript-editing tools are not included. The archived final seed and published robustness seeds remain in the package because they are needed for reproducibility; this does not imply that the final seed was prespecified.

## Requirements and installation

The release was tested locally on macOS with Python 3.12 and the following package versions, specified in [requirements.txt](requirements.txt):

- NumPy 2.0.2 and Pandas 2.2.3
- Matplotlib 3.9.4 and SciPy 1.13.1
- Scikit-Learn 1.5.2 and PyTorch 2.8.0

The original study computations were performed on the University of Georgia cluster using AMD EPYC 9534 computing nodes and PyTorch 2.8.0, with four intra-operation threads, one inter-operation thread, and deterministic algorithms enabled. The implementation uses CPU computation.

```bash
git clone https://github.com/statszyc/SERS_decomp_NN.git
cd SERS_decomp_NN
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python unpack_data.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Data, results, and approved figures are distributed as three ZIP archives to keep the repository compact. `unpack_data.py` checks their SHA-256 digests and extracts them into `data/`, `results/`, and `figures/` without replacing different existing files.

## Reproduce the saved results

```bash
python -m unittest discover -s tests
python -m sers.reproduce
python -m sers.plot_results
```

Saved-result reproduction and plotting require no model fitting; the test suite includes tiny synthetic fits to check execution. These commands verify the complete primary candidate matrices, reproduce both selected configurations and all 70 reconstruction-loss comparator choices, export tables and summaries, and replot the numerical panels under `reproduced/`. The numerical plotting code produces compact, editable Matplotlib plots; it does not claim pixel-identical reproduction of the publication layouts. The approved publication images are supplied separately in `figures/`.

The expected selections are:

| Analysis | Configuration ID | R | s | lambda_neg | lambda_flat | Learning rate | Entry fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All 37 validation windows | h64revised_theta_01308 | 5 | 12.992632226094093 | 1 | 0.0001 | 0.01 | 2882/7560 |
| Six pyocyanin experimental windows | h64revised_theta_02018 | 7 | 12.992632226094093 | 100 | 0.0001 | 0.0001 | 1042/7560 |

Both minimum entry fractions are unique. Hidden size is fixed at H = 64 in the primary grid and is not a searched grid coordinate. Machine-readable values retain their original precision.

## Try your own processed spectra

```bash
python -m sers.fit_window --demo --output reproduced/demo
python -m sers.fit_window --input my_window.csv --levels 0 1 2 --output reproduced/my_window
```

The demo exports a complete CSV input template, fitted curves, coefficients,
metrics, and a recovery plot. The custom-data interface uses the fixed published
configuration; an optional reference is used only after fitting. See
[examples/README.md](examples/README.md) for the input schema and output
conventions. Add `--epochs 2` only for a quick execution check; the default is
the paper's 1,000 epochs. A successful smoke test is not evidence of recovery
quality.

## Run the final models

Each command below runs one task. Task indices and identities are listed in `data/validation_index.json`, `data/test_index.json`, and `data/window_size_index.json`.

```bash
# Primary selected model and the test-window loss comparator
python -m sers.replay --split validation --task 0 --output reproduced/validation_0.json
python -m sers.replay --split test --task 0 --candidate loss --output reproduced/test_loss_0.json

# Conventional and prior neural comparators
python -m sers.replay --method mcrals --task 0 --output reproduced/mcrals_0.json
python -m sers.replay --method prior --task 0 --output reproduced/prior_0.json

# Published post-selection sensitivity analyses
python -m sers.replay --analysis robustness --split test --task 0 --base-seed 18 --output reproduced/robustness_0.json
python -m sers.replay --analysis hidden --task 0 --hidden-size 32 --base-seed 18 --output reproduced/hidden_0.json
python -m sers.replay --analysis window --task 0 --output reproduced/window_0.json
python -m sers.ablation --panel B --output reproduced/ablation_B.json
```

Training uses 1,000 epochs. Saved-result reproduction and plotting use the archived outputs without retraining. Commands, computation environments, and release checks are described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

To evaluate an explicit chunk of the final grid, use `--grid-start` and `--grid-stop` (exclusive), for example `--grid-start 0 --grid-stop 48`. A full grid contains 7,560 fits **per window**; the primary validation and test searches total 529,200 fits. No full-grid training is launched by default.

## Procedure and selection boundaries

1. Load a frozen processed window. The reference spectrum is not used to train network weights; it is used only to evaluate recovery after fitting.
2. Fit the two branches using the archived task seed. All candidates within a task share that seed.
3. For validation, compute recovery scores after fitting and rank candidates by decreasing cosine similarity. Exact score ties are ordered by candidate ID.
4. For candidate m, compute `p_entry(m) = max_i rank_i(m) / M` across the validation windows. Select the candidate with the smallest entry fraction.
5. In each held-out window, fit the fixed validation-selected configuration. The window-specific loss comparator minimizes the pure reconstruction residual over the same candidate grid, with candidate-ID tie breaking. It does not minimize the regularized total training loss and does not use the reference spectrum for selection.
6. Evaluate recovery against the reference only after the fit and comparator choice.

The primary base seed is 710. Task seeds use `numpy.random.SeedSequence([710, domain_local_task_index, 0])`. The published robustness analysis uses base seeds 18, 42, and 251 with namespace 20260810. These are three fixed post-selection evaluations, not a search over seeds. The hidden-size analysis uses the same matched-seed namespace and does not change the primary H = 64 setting.

## Data and figure guide

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for fields, units, precision, and input boundaries, and [FIGURE_TABLE_MAP.md](FIGURE_TABLE_MAP.md) for the file-to-figure/table mapping.

The supplied inputs are the exact processed arrays consumed by the final computations. They are not a complete raw-instrument archive. Experimental preprocessing and simulation-source reconstruction should not be inferred from the frozen arrays. `sliding_windows` demonstrates window construction from already-processed spectra; exact replays use the supplied frozen windows, including the original replicate-subset construction.

The original figure data and model output amplitudes are distinguished from display-normalized curves. Figure 3A is data-driven and is numerically generated with B–E. S3 includes the original display-background curves as well as the before/after recovery comparison. S5/S9 include their zoom panels. S10 plots the median across validation windows for base seed 42, not a mean or a multi-seed uncertainty band. Figure 2 and Figure S1 are schematic artwork, not numerical outputs requiring a training or plotting script; the conceptual network diagrams in Figure 1A and Figure S2 are also archived separately.

## Attribution and reuse

Authors: Yingchuan Zhang, Haoran Lu, Yanjun Yang, Jiaheng Cui, Xinyi Liu, Ping Ma, Wenxuan Zhong, and Yiping Zhao.

Please cite the accompanying manuscript when using these materials. A publication DOI will be added when available. Third-party Python dependencies retain their own licenses. No additional software or data license is granted by this release; contact the authors to establish reuse permissions where required. Repository issues can be used for technical questions.
