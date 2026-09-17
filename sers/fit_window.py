"""Fit one window of the user's already-processed spectra with fixed theta*.

Example:
  python -m sers.fit_window --demo --output reproduced/demo
  python -m sers.fit_window --input window.csv --levels 0 1 2 --output reproduced/my_window

The optional reference is used only for post-fit evaluation. No grid or seed
search is performed. This interface does not preprocess raw instrument spectra.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io import grid
from .training import run_single_trial
from .features import compute_ground_truth_metrics

SELECTED_ID = 'h64revised_theta_01308'


def demo_frame():
    """Synthetic input template, not a manuscript dataset or claimed result."""
    x = np.linspace(400, 1700, 96)
    background = .3 + .12*np.exp(-((x-900)/250)**2)
    analyte = .02 + np.exp(-((x-1000)/50)**2) + .6*np.exp(-((x-1450)/70)**2)
    frame = pd.DataFrame({'wavenumber': x, 'background': background})
    for j, (a, b) in enumerate(zip([.15, .4, .8], [.9, .8, .7])):
        frame[f'mix_{j}'] = a*analyte + b*background
    frame['reference'] = analyte
    return frame


def parse_window(frame, levels):
    """Validate a common axis, one fixed background and ordered input columns."""
    mixes = [c for c in frame.columns if c.startswith('mix_')]
    expected = [f'mix_{j}' for j in range(len(mixes))]
    if mixes != expected or len(mixes) < 2:
        raise ValueError('Use consecutive mix_0, mix_1, ... columns (at least two)')
    required = {'wavenumber', 'background', *mixes}
    if not required.issubset(frame.columns):
        raise ValueError('Missing wavenumber or background column')
    if set(frame.columns) - required - {'reference'}:
        raise ValueError('Unexpected columns; use the documented CSV schema')
    values = frame.to_numpy(dtype=float)
    if len(frame) < 8 or not np.isfinite(values).all():
        raise ValueError('At least eight finite spectral points are required')
    x = frame.wavenumber.to_numpy(float)
    if not np.all(np.diff(x) > 0):
        raise ValueError('wavenumber must be strictly increasing')
    levels = np.asarray(levels, float)
    if levels.shape != (len(mixes),) or not np.isfinite(levels).all() or not np.all(np.diff(levels) > 0):
        raise ValueError('Provide one strictly increasing ordered level per mixture')
    mixtures = frame[mixes].to_numpy(float).T
    background = frame.background.to_numpy(float)
    if np.linalg.norm(background) == 0 or np.linalg.norm(mixtures) == 0:
        raise ValueError('Background and mixtures must not be all zero')
    reference = frame.reference.to_numpy(float) if 'reference' in frame else None
    if reference is not None and np.linalg.norm(reference) == 0:
        raise ValueError('Optional reference must not be all zero')
    task = dict(x_axis=x.tolist(), x_mix=mixtures.tolist(), concentrations=levels.tolist(),
                bg=[background.tolist()], gt_f=None, metadata={'preprocess_mode': 'none'},
                dataset_id='user_processed', task_id='user_window')
    return task, reference


def fit_window(frame, levels, epochs=1000, seed=710):
    """Use the published fixed configuration; reference never reaches training."""
    if epochs < 1:
        raise ValueError('epochs must be positive')
    task, reference = parse_window(frame, levels)
    theta = next(row['theta'] for row in grid() if row['theta_id'] == SELECTED_ID)
    # Validate in the arithmetic actually consumed by the unchanged runner.
    # Float64-valid CSV values can overflow or collapse on conversion to float32.
    with np.errstate(over='ignore', invalid='ignore'):
        x32 = np.asarray(task['x_axis'], dtype=np.float32)
        levels32 = np.asarray(task['concentrations'], dtype=np.float32)
        bg32 = np.asarray(task['bg'][0], dtype=np.float32)*theta['scale_factor']
        mixtures32 = np.asarray(task['x_mix'], dtype=np.float32)*theta['scale_factor']
    if not all(np.isfinite(a).all() for a in [x32, levels32, bg32, mixtures32]):
        raise ValueError('Inputs must remain finite at the model float32 precision and scale')
    if not np.all(np.diff(x32) > 0) or not np.all(np.diff(levels32) > 0):
        raise ValueError('Axis and ordered levels must remain distinct and increasing in float32')
    background_area = np.trapezoid(bg32, x32)
    if not np.isfinite(background_area) or abs(background_area) < 1e-12:
        raise ValueError('The model area convention requires a nonzero signed background area')
    result = run_single_trial(task, theta, SELECTED_ID, epochs=epochs, device='cpu',
                              torch_seed=seed, return_artifacts=True)
    artifacts = result['artifacts']
    recovered = np.asarray(artifacts['f_final'])/theta['scale_factor']
    coeff = np.asarray(artifacts['g_final'])
    bg_coeff = np.asarray(artifacts['background_coeffs'])[:, 0]
    recon = np.asarray(artifacts['recon_matrix'])
    if (not all(np.isfinite(a).all() for a in [recovered, coeff, bg_coeff, recon])
            or np.linalg.norm(recovered) == 0):
        raise ValueError('Fit produced nonfinite or zero recovered output; no result exported')
    rebuilt = coeff[:, None]*recovered + bg_coeff[:, None]*np.asarray(task['bg'][0])
    if not np.allclose(rebuilt, recon, rtol=2e-5, atol=2e-6):
        raise ValueError('Unstable area rescaling: exported factors do not reconstruct the fit')
    result['postfit_reference_metrics'] = compute_ground_truth_metrics(
        None if reference is None else reference.astype(np.float32)*theta['scale_factor'],
        np.asarray(result['artifacts']['f_final'], dtype=np.float32))
    result['execution'] = {'seed': seed, 'configuration_id': SELECTED_ID,
                           'purpose': 'custom-data example',
                           'reference_used_for_training': False}
    return result


def save_outputs(frame, levels, result, out):
    """Export raw model-scale outputs and a clearly separated display plot."""
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out/'input.csv', index=False)
    (out/'fit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    artifacts = result['artifacts']
    # Cancel the numerical training scale, not the area convention of the
    # decomposition. g_final and the background coefficients pair with it.
    recovered = np.asarray(artifacts['f_final'])/result['theta']['scale_factor']
    curves = pd.DataFrame({'wavenumber': frame.wavenumber, 'recovered_analyte': recovered})
    for j, curve in enumerate(artifacts['recon_matrix']):
        curves[f'reconstructed_mix_{j}'] = curve
    curves.to_csv(out/'curves.csv', index=False)
    coefficients = pd.DataFrame({
        'ordered_level': levels, 'analyte_coefficient': artifacts['g_final'],
        'background_coefficient': np.asarray(artifacts['background_coeffs'])[:, 0]})
    coefficients.to_csv(out/'coefficients.csv', index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    x = frame.wavenumber
    for column in [c for c in frame if c.startswith('mix_')]:
        axs[0].plot(x, frame[column], label=column, lw=1)
    axs[0].plot(x, frame.background, label='Background', color='black', lw=1)
    axs[0].set(title='Already-processed inputs', ylabel='Input intensity')
    axs[1].plot(x, recovered/np.linalg.norm(recovered), label='Recovered', color='#D92323')
    if 'reference' in frame:
        reference = frame.reference.to_numpy()
        axs[1].plot(x, reference/np.linalg.norm(reference), label='Reference (post-fit only)',
                    color='black', lw=1)
    axs[1].set(title='Recovered analyte shape', ylabel='Unit-L2 display intensity')
    for ax in axs:
        ax.set_xlabel(r'Raman shift (cm$^{-1}$)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'recovery.png', dpi=160)
    fig.savefig(out/'recovery.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--demo', action='store_true')
    parser.add_argument('--levels', nargs='+', type=float, help='One ordered coordinate per mixture')
    parser.add_argument('--epochs', type=int, default=1000, help='Use 2 only for an execution smoke test')
    parser.add_argument('--seed', type=int, default=710, help='One initialization seed, not a search')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory exists; choose a new path')
    if args.demo and args.levels is not None:
        parser.error('The synthetic demo supplies its own levels 0, 1, 2')
    if not args.demo and args.levels is None:
        parser.error('--levels is required with --input')
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be a nonnegative 32-bit integer')
    levels = [0, 1, 2] if args.demo else args.levels
    frame = demo_frame() if args.demo else pd.read_csv(args.input)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        result = fit_window(frame, levels, args.epochs, args.seed)
        save_outputs(frame, levels, result, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Saved fit, curves, coefficients and plots to {args.output}')
    if args.epochs != 1000:
        print('Nonstandard epoch count: this is not a paper-protocol training run.')


if __name__ == '__main__':
    main()
