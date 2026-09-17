# Reproduction scope and known training limitation

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

## Fitting again

The public replay entry points run the original model and fixed tasks. The
following are different claims:

1. An entry point runs and produces finite results.
2. A new fit agrees with the historical cluster result.

The first was checked locally for the primary model, loss-comparator fit,
MCR-ALS, prior network, robustness, hidden-size, window-size and Figure 1B
ablation entry points. A full 529,200-fit primary campaign was not rerun.
The second does **not** hold for all sampled fits in the local environment.

### Open issue: Figure 1B Fourier ablation

| Result | Recovery q |
| --- | --- |
| Archived cluster result | -0.4595472384187544 |
| Tested local replay | 0.6877471208572388 |

The same frozen input arrays, configuration, initialization seed and 1,000
epochs were used. The collected original runner and the public runner produced
bitwise-identical local curves; a repeat local run also matched. Thus that
comparison does not identify a numerical change introduced by the packaging.
It also does not establish why the local result differs from the archive.

CPU family, BLAS and PyTorch build can alter a nonconvex optimization trajectory,
but attributing this specific discrepancy to them requires a controlled replay
in the historical cluster environment. That check is outstanding. Do not use
the successful local execution as a claim that the archived ablation has been
retrained exactly. The original archived curve and q have not been replaced.

Other sampled neural replays also differed, generally less substantially; this
is not an issue confined to decimal rounding. Saved-result plots do not depend
on these new fits. The local checks used macOS with NumPy 2.0.2, SciPy 1.13.1
and PyTorch 2.8.0; the primary cluster runs used AMD EPYC 9534 CPUs. The pinned
requirements are a tested release environment, not a complete historical
environment export. Original NumPy `trapz` calls emit deprecation warnings with
NumPy 2; these calls remain unchanged to preserve the training implementation.

## Custom-data example

`sers.fit_window` is a convenience interface for one processed window, not a
rerun of the paper dataset or an optimized configuration for new data. It calls
the unchanged training function using the frozen global theta*. The optional
reference is withheld until post-fit evaluation. Its synthetic demo and
two-epoch test runs are execution examples, not scientific validation evidence.
