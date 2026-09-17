# Reproduction guide

## Saved numerical results

`python -m sers.reproduce` reconstructs the two consensus selections, all
70 loss-comparator choices, and tables from the archived numerical inputs.
`python -m sers.plot_results` redraws their numerical content without training.
This path is verified against the released matrices and figure sources.
The approved publication images are archived separately; pixel-identical
typography is not the contract of the compact Matplotlib renderer.

The plotting supplement adds Figure 3A's data-driven illustration, S3 overview
backgrounds, S5/S9 zooms, first-entry visibility and final labels in Figure 4.
It restores the original Figure 1 area-based display transform and the S10
base-seed-42 window-median statistic. It changes no archived scores, selected
configurations, model/training implementation, or previously distributed ZIP
contents.

## Computation environments

The original study computations used AMD EPYC 9534 computing nodes on the
University of Georgia cluster with PyTorch 2.8.0, four intra-operation threads,
one inter-operation thread, and deterministic algorithms enabled.

Local release checks used macOS, Python 3.12, NumPy 2.0.2, Pandas 2.2.3,
Matplotlib 3.9.4, SciPy 1.13.1, Scikit-Learn 1.5.2, and PyTorch 2.8.0.
These package versions are specified in `requirements.txt`.

## Training commands and release checks

`python -m sers.replay` fits the original model on a supplied frozen task.
`python -m sers.ablation` runs the representative ablation fits. Both use
1,000 epochs; command examples are provided in the README.

Release checks covered archived-result reconstruction, numerical plotting,
and sampled execution of the primary model, loss comparator, MCR-ALS, prior
network, robustness, hidden-size, window-size, and ablation entry points.
The complete check summary is in
[`metadata/supplement_verification.json`](metadata/supplement_verification.json).
Recorded training-check measurements are in
[`metadata/training_replay_check.json`](metadata/training_replay_check.json).

## Custom-data example

`sers.fit_window` is a convenience interface for one processed window, not a
rerun of the paper dataset or an optimized configuration for new data. It calls
the unchanged training function using the frozen global theta*. The optional
reference is withheld until post-fit evaluation. Its synthetic demo and
two-epoch test runs are execution examples, not scientific validation evidence.
