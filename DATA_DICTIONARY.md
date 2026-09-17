# Data dictionary

## Frozen inputs

`data/*_tasks.jsonl.gz` contains one JSON object per window. The JSON index files expose identities and metadata without loading the spectra.

| Field | Meaning |
| --- | --- |
| task_id | Frozen window identity. Historical identifiers containing `validation` can belong to the held-out test split; use the enclosing file and metadata. |
| dataset_id | Dataset family; virus identity is also stored in metadata. |
| x_axis | Raman-shift coordinates in cm^-1. |
| x_mix | Processed mixture intensities, shape K by spectral points. |
| concentrations | Historical field name for the ordered/relative coordinates actually used by the model. Do not reinterpret all values as measured concentrations. |
| bg | Supplied fixed background spectra, one row per background. |
| gt_f | Reference analyte spectrum for post-fitting evaluation; removed before fitting by the replay interface. |
| metadata | Dataset identity, window positions, and retained preprocessing/replicate-subset metadata. Personal filesystem paths have been removed. |

Validation comprises DNA/RNA experimental and simulated data (9 windows each), pyocyanin experimental data (6 windows), and pyocyanin simulated data (13 windows). Test data comprise Ad5, FluB, and HMPV-A (8 windows each), and HMPV-B (9 windows). All primary windows have K = 3. The 92 supplementary K = 5/7 windows are distributed separately and were not used to select the primary configuration.

Experimental replicate-subset metadata is retained because these frozen windows must not be replaced by means calculated from a different subset. The unitless model input coordinate is normalized to [0, 1] inside the training code. No physical calibration is inferred from a task filename.

The five displayed pyocyanin simulated levels in Figure S3 correspond, in display order, to unitless pyocyanin coefficients 0.02, 0.08, 0.15, 0.30, and 0.48. The corresponding background weights are 1 minus these coefficients. The full coefficient set is 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, and 0.48. Source labels and display ordering are retained; historical date-like labels should not be read as literal dates.

The five displayed pyocyanin experimental levels correspond to source files 9, 7, 5, 4, and 1, respectively, with concentrations 2.60e-4, 1.04e-3, 4.17e-3, 8.33e-3, and 6.67e-2 mg/mL. These rounded explanatory values do not replace the exact model coordinates. Virus concentration labels, where explicitly presented as physical concentrations, are in PFU/mL.

## Configurations

`data/grid.json` is ordered to match the columns of all released score/residual matrices. It records all 7,560 configurations at full machine precision. Legacy IDs are retained as stable identifiers, not interpreted as contiguous integers.

| Code field | Manuscript quantity |
| --- | --- |
| num_frequencies | Fourier-feature order R |
| scale_factor | Numerical scale s |
| lambda_penalty | Negative-intensity penalty weight lambda_neg |
| lambda_flat | Flatness penalty weight lambda_flat |
| lr | Adam learning rate eta |
| hidden_dim, z_dim | Fixed primary branch sizes H = 64 |

## Result matrices

`results/validation_scores.npz` stores `q`, `rank_position`, and `nonnegative_fraction`, all indexed by its `task_ids` and `theta_ids`. Ranks are one-based. Recovery q is the cosine similarity to the operational reference, not a training-loss term.

`results/validation_residuals.npz` and `results/test_residuals.npz` store the complete finite `residual_norm_ratio` matrices, matching task/candidate IDs, and task seeds. The metric is `norm(x_mix - reconstruction) / (norm(x_mix) + 1e-12)`. It is distinct from regularized `final_loss`.

`validation_selected_vs_loss.csv` and `test_scores.csv` contain the paired final recovery results. Test field `global_q` denotes recovery by the fixed validation-selected configuration. Delta fields subtract the loss-comparator q from the selected q.

Supporting CSVs retain full computational precision. Additional columns such as runtime, reconstruction residual, Pearson correlation, and R-squared are auxiliary diagnostics and are not substituted for q. Hidden-size summary confidence intervals are archived results; the quick reproduction script recomputes descriptive summaries, not those bootstrap intervals.

## Plotted values and numerical precision

`data/figures/` contains figure-specific data. Columns named `display_*`, `plotted_intensity`, `normalized_intensity`, or `scaled_intensity_offset` are display-transformed values, not raw intensities. In Figure S3C, `source_intensity` is an earlier display-scale value; the exact model inputs remain in the frozen task file.

The Figure 3 numerical source contains a compressed display coordinate for panel B. `source_y` retains the uncompressed normalized ranks. The compact release replot uses those uncompressed ranks. Publication images retain the approved axis treatment.

Training and some recovery calculations used float32. Recomputing cosine similarity from decimal CSV curves with float64 may produce differences of order 1e-7. The archived primary score matrix is the authority for exact candidate ranking.
