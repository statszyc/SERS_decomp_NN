# Fit your own processed window

Start with a runnable synthetic example (not a manuscript result):

```bash
python -m sers.fit_window --demo --output reproduced/demo
```

The output `input.csv` is also a complete input template. To test execution
quickly, add `--epochs 2` and use a different output directory; that abbreviated
fit is not a meaningful recovery result. The default is 1,000 epochs.

For your data, prepare a CSV with one spectral point per row:

| wavenumber | background | mix_0 | mix_1 | mix_2 | reference |
| --- | --- | --- | --- | --- | --- |
| Raman shift | supplied fixed background | first mixture | second mixture | third mixture | optional evaluation spectrum |

Use at least eight strictly increasing Raman-shift coordinates in cm^-1.
All columns must share that axis, have finite numeric values, and already be
processed on the intended consistent intensity scale. Column `reference` is
optional. Supply at least two mixtures, named consecutively `mix_0`, `mix_1`, …
in increasing level order; the primary paper uses three. This example supports
one fixed background, matching the primary method. It does not perform raw-data
preprocessing, physical concentration calibration, or reference construction.
The background must have nonzero signed area because the original model uses
that area to fix the factorization scale. Axis and ordered-level values must
remain distinct in float32; excessively large offsets can erase small
differences at training precision. Degenerate or nonfinite fits are rejected,
not exported as valid recovery results.

```bash
python -m sers.fit_window --input my_window.csv --levels 0 1 2 --output reproduced/my_window
```

`--levels` supplies the ordered/relative coordinate of each mixture, normalized
within the window by the existing training implementation. These values are not
inferred from column names or assumed to be measured concentrations. Choose them
to represent your processed dataset; use the frozen tasks for paper replays.

The fit uses the published fixed configuration `h64revised_theta_01308`, not a
new configuration selected for your data. It uses one explicit initialization
seed (default 710). No reference-based tuning, grid search, or seed search is
performed. A reference, if provided, is evaluated only after fitting; without it,
recovery q is unavailable but reconstruction diagnostics are still exported.
Transfer performance on new datasets is not guaranteed.

## Outputs

- `fit.json`: configuration, seed, training and reconstruction metrics, optional
  post-fit reference metrics, and the original model artifacts.
- `curves.csv`: recovered analyte and reconstructed mixtures. The numerical
  training scale `s` is removed; the original area-based factorization convention
  is preserved.
- `coefficients.csv`: area-adjusted analyte coefficient (`g_final`) and supplied
  background coefficient for each ordered level. These are decomposition
  coefficients, not calibrated concentrations.
- `recovery.png` / `recovery.pdf`: input curves and the recovered shape; the
  right-hand display uses independent unit-L2 normalization, not raw amplitudes.
- `input.csv`: the input actually used.

In the normal nonzero-area case,
`reconstructed_mix_i ≈ analyte_coefficient_i * recovered_analyte +
background_coefficient_i * background`, up to float32 rounding. `fit.json`
also retains the unadjusted branch coefficients for inspection. Output folders
must not already exist, preventing accidental replacement of earlier fits.
