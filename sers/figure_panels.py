"""Data-driven panels and display transforms used by the public replots.

No training or configuration selection is performed on new data here. Selection
is reconstructed only from the archived validation score matrix.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .io import ROOT, tasks
from .selection import consensus

RED = '#D92323'
GREEN = '#007A3D'
BLUE = '#1558A6'
GOLD = '#D4AC0D'
PYO_SYMBOL = r'$\theta^*_{\mathrm{Pyo}}$'
W5_ID = 'pyocyanin_experimental_task_04_06__perturb_fold_00'
VALIDATION_DATASETS = [
    'pyocyanin_experimental', 'pyocyanin_simulated_full',
    'dnarna_experimental', 'dnarna_simulated_full',
]
TEST_DATASETS = ['Ad5', 'FluB', 'HMPVA', 'HMPVB']
DISPLAY_NAMES = {
    'pyocyanin_experimental': 'Pyocyanin experimental',
    'pyocyanin_simulated_full': 'Pyocyanin simulated',
    'dnarna_experimental': 'DNA/RNA experimental',
    'dnarna_simulated_full': 'DNA/RNA simulated',
    'HMPVA': 'HMPV-A', 'HMPVB': 'HMPV-B',
}


def unit_area(x, y):
    """Original signed-trapezoidal display normalization, with area fallback."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or not np.isfinite(y).all():
        raise ValueError('Finite curves on a shared Raman-shift axis are required')
    area = np.trapezoid(y, x)
    if not np.isfinite(area) or abs(area) < 1e-12:
        area = np.trapezoid(np.abs(y), x)
    if not np.isfinite(area) or abs(area) < 1e-12:
        raise ValueError('Cannot area-normalize a zero curve')
    return y / area


def figure1_display(payload):
    reference = unit_area(payload['x_axis'], payload['reference_curve'])
    peak = np.max(np.abs(reference))
    return {
        key: unit_area(payload['x_axis'], payload[key]) / peak
        for key in ['reference_curve', 'retained_curve', 'ablated_curve']
    }


def plot_figure1():
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for panel, ax in zip('BCDE', axs.flat):
        with np.load(ROOT / f'data/figures/figure1_{panel}.npz') as payload:
            curves = figure1_display(payload)
            for key, label, color in [('reference_curve', 'Ref.', 'black'),
                                      ('retained_curve', 'Retained', GREEN),
                                      ('ablated_curve', 'Ablated', RED)]:
                ax.plot(payload['x_axis'], curves[key], label=label, color=color, lw=1)
        ax.set(title=f'Figure 1{panel}', xlim=(450, 1700), ylim=(-.58, 1.85),
               xlabel=r'Raman shift (cm$^{-1}$)', ylabel='Display intensity (a.u.)')
        ax.legend(fontsize=8)
    return fig


def pyocyanin_selection():
    with np.load(ROOT / 'results/validation_scores.npz') as payload:
        ids = payload['theta_ids']
        mask = np.char.startswith(payload['task_ids'], 'pyocyanin_experimental')
        q = payload['q'][mask]
        task_ids = payload['task_ids'][mask]
    ranks, entry, order = consensus(q, ids)
    if len(task_ids) != 6:
        raise ValueError('Expected six frozen pyocyanin windows')
    return dict(ids=ids, task_ids=task_ids, q=q, ranks=ranks, entry=entry, order=order)


def plot_figure3():
    """A: W5 inputs, retention, consensus; B–E: archived numerical results."""
    data = pyocyanin_selection()
    ids, ranks, entry, order = (data[k] for k in ['ids', 'ranks', 'entry', 'order'])
    m = len(ids)
    selected = order[0]
    wi = list(data['task_ids']).index(W5_ID)
    task = next(t for t in tasks('validation') if t['task_id'] == W5_ID)
    fig = plt.figure(figsize=(13, 11), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1])
    top = grid[0, :].subgridspec(1, 3, width_ratios=[1.15, 1, 1.1], wspace=.24)
    ax_input, ax_rank, ax_consensus = [fig.add_subplot(top[0, j]) for j in range(3)]
    x = np.asarray(task['x_axis'])
    # A common positive scale preserves relative input amplitudes.
    mix, bg = np.asarray(task['x_mix']), np.asarray(task['bg'][0])
    scale = np.max(np.abs(mix))
    for j, curve in enumerate(mix):
        ax_input.plot(x, curve / scale, lw=1, label=rf'$I^{{\mathrm{{mix}}}}_{{{j+5}}}$')
    ax_input.plot(x, bg / scale, color='black', ls='--', lw=1, label='Background')
    ax_input.set(title=r'A  $W_5$: input triplet', xlabel=r'Raman shift (cm$^{-1}$)',
                 ylabel='Intensity / input maximum', xlim=(450, 1700))
    ax_input.legend(fontsize=8, ncol=2)
    sorted_q = np.sort(data['q'][wi])[::-1]
    positions = np.arange(1, m + 1) / m
    ax_rank.plot(positions, sorted_q, color='0.35', lw=1)
    rank = ranks[wi, selected] / m
    score = data['q'][wi, selected]
    ax_rank.axvspan(0, .1, color=BLUE, alpha=.13, label='Top 10%')
    ax_rank.axvspan(.1, .2, color=BLUE, alpha=.06, label='Top 10–20%')
    ax_rank.scatter([rank], [score], color=RED, zorder=4)
    ax_rank.text(.97, .32, PYO_SYMBOL + f': rank = {rank:.5f}',
                 color=RED, transform=ax_rank.transAxes, ha='right', fontsize=10)
    ax_rank.text(.97, .21, f'q = {score:.5f}; retained at 20%',
                 transform=ax_rank.transAxes, ha='right', fontsize=9)
    ax_rank.set(title=r'Within $W_5$', xlabel='Normalized rank', ylabel='Recovery score q',
                xlim=(0, .3))
    ax_rank.set_ylim(sorted_q[int(.3*m)] - .015, sorted_q[0] + .005)
    ax_rank.legend(fontsize=8, loc='upper right')
    windows = np.arange(1, 7)
    normalized = ranks[:, selected] / m
    ax_consensus.plot(windows, normalized, '-o', color=RED, label=PYO_SYMBOL)
    ax_consensus.axhline(entry[selected], color='black', ls=':', lw=1)
    ax_consensus.text(.03, .95, r'$p_{\mathrm{entry}}=\max_i r_i/M$'
                      + f'\n= {entry[selected]:.5f}', va='top', fontsize=10,
                      transform=ax_consensus.transAxes)
    ax_consensus.set(title='Across six windows', xlabel='Window',
                     ylabel='Normalized rank', xticks=windows, ylim=(0, .2))
    ax_consensus.legend(fontsize=9, loc='lower right')
    ax_b, ax_c = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    colors = [RED, '#6464B8', '#55A6A6', '#804580']
    for j, color in enumerate(colors):
        label = PYO_SYMBOL + r' ($m_1$)' if j == 0 else rf'Candidate $m_{j+1}$'
        ax_b.plot(windows, ranks[:, order[j]] / m, '-o', color=color, label=label)
    ax_b.set(title='B  Candidate rank trajectories', xlabel='Window',
             ylabel='Normalized rank (smaller is better)', xticks=windows)
    ax_b.invert_yaxis()
    ax_b.legend(fontsize=9, ncol=2)
    values, counts = np.unique(entry, return_counts=True)
    ax_c.step(np.r_[0, values], np.r_[0, np.cumsum(counts)], where='post', color=BLUE)
    for j in range(4):
        threshold = entry[order[j]]
        count = np.count_nonzero(entry <= threshold)
        ax_c.scatter([threshold], [count], color=colors[j], zorder=4)
        ax_c.annotate(rf'$p_{{\mathrm{{entry}}}}(m_{j+1})={threshold:.5f}$',
                      (threshold, count), xytext=(-8, 8), textcoords='offset points',
                      ha='right', fontsize=9, color=colors[j])
    ax_c.set(title='C  Common retained set', xlabel='Retained fraction p',
             ylabel='Common-set size', xlim=(entry.min()-.012, entry[order[3]]+.003),
             ylim=(-.1, 5))
    ax_d, ax_e = fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])
    source = pd.read_csv(ROOT / 'data/figures/figure3.csv')
    scores = source[source.panel == 'D']
    series = list(scores.series.unique())
    labels = ['Loss Comp.' if 'Loss' in s else 'Oracle' if s == 'Oracle' else PYO_SYMBOL
              for s in series]
    palette = [GREEN if 'Loss' in s else GOLD if s == 'Oracle' else RED for s in series]
    boxes = ax_d.boxplot([scores[scores.series == s].source_y for s in series],
                         tick_labels=labels, patch_artist=True, showmeans=True)
    for box, color in zip(boxes['boxes'], palette):
        box.set(facecolor=color, alpha=.3)
    ax_d.set(title='D  Six-window recovery', ylabel='Recovery score q')
    for key, part in source[source.panel == 'E'].groupby('series', sort=False):
        label = 'Ref.' if key == 'Reference' else 'Loss Comp.' if 'Loss' in key else 'Selected'
        color = 'black' if label == 'Ref.' else GREEN if label == 'Loss Comp.' else RED
        ax_e.plot(part.x, part.display_y, color=color, label=label, lw=1)
    ax_e.set(title=r'E  $W_5$ recovery', xlabel=r'Raman shift (cm$^{-1}$)',
             ylabel='Display intensity (a.u.)')
    ax_e.legend(fontsize=9)
    return fig


def plot_consensus_entry(ax, entry):
    """Show the first-entry region so selection is not hidden by 7,560 entries."""
    values, counts = np.unique(entry, return_counts=True)
    totals = np.cumsum(counts)
    pstar = float(entry.min())
    visible = entry <= pstar + .04
    ymax = max(5, np.count_nonzero(visible) + 1)
    ax.step(np.r_[0, values], np.r_[0, totals], where='post', color=BLUE)
    ax.axvline(pstar, color=RED, lw=1)
    ax.scatter([pstar], [np.count_nonzero(entry == pstar)], color=RED, zorder=4)
    ax.text(.30, .94, rf'$p^*={pstar:.5f}$' + '\nFirst common entry: selected',
            transform=ax.transAxes, va='top', color=RED, fontsize=9)
    ax.set(xlim=(pstar-.01, pstar+.04), ylim=(0, ymax), xlabel='Retained fraction p',
           ylabel='Common-set size', title='Figure 4A: first-entry region')


def s3_backgrounds():
    return pd.read_csv(ROOT / 'plot_inputs/figureS3_backgrounds.csv')


def plot_s3_overview():
    overview = pd.read_csv(ROOT / 'data/figures/figureS3_overview.csv')
    backgrounds = s3_backgrounds()
    datasets = VALIDATION_DATASETS + TEST_DATASETS
    fig, axs = plt.subplots(4, 2, figsize=(12, 12))
    level_colors = ['#0000A0', '#008080', '#808000', '#FF8000', '#FF0000']
    for dataset, ax in zip(datasets, axs.flat):
        sub = overview[overview.dataset_id == dataset]
        if sub.empty:
            raise ValueError(f'Missing overview dataset: {dataset}')
        for level, part in sub.groupby('display_order', sort=True):
            ax.plot(part.wavenumber, part.scaled_intensity_offset,
                    color=level_colors[int(level)-1], label=f'Level {level}', lw=1)
        bg = backgrounds[backgrounds.dataset_id == dataset]
        ax.plot(bg.wavenumber, bg.display_intensity, '--', color='black',
                lw=1, label='Background')
        ax.set(title=DISPLAY_NAMES.get(dataset, dataset), xlim=(450, 1700),
               ylim=(-1.15, 8.35), xlabel=r'Raman shift (cm$^{-1}$)',
               ylabel='Offset display intensity')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .97))
    return fig


def diagnostic_figure(df, number, method, zoom):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    names = {'selected': 'Selected', 'reference': 'Ref.', 'mcrals': 'MCR-ALS',
             'prior_dual_network': 'Prior dual-network'}
    x = df['raman_shift_cm-1']
    for column, ax in enumerate(axs[0]):
        for key, color in [('reference', 'black'), ('selected', BLUE), (method, RED)]:
            ax.plot(x, df[key], color=color, label=names[key], lw=1)
        ax.set(title=f'Figure S{number}' + ('B: peak zoom' if column else 'A: spectrum'),
               ylabel='Display intensity')
    for column, ax in enumerate(axs[1]):
        for key, color in [('selected', BLUE), (method, RED)]:
            ax.plot(x, df[key+'_absolute_deviation'], color=color, label=names[key], lw=1)
        ax.set(title=f'Figure S{number}' + ('D: deviation zoom' if column else 'C: deviation'),
               ylabel='Absolute deviation')
    for row in range(2):
        ax = axs[row, 1]
        ax.set_xlim(*zoom)
        mask = x.between(*zoom)
        fields = ['reference', 'selected', method] if row == 0 else [
            'selected_absolute_deviation', method+'_absolute_deviation']
        low, high = df.loc[mask, fields].min().min(), df.loc[mask, fields].max().max()
        pad = max((high-low)*.07, 1e-6)
        ax.set_ylim(low-pad, high+pad)
    for ax in axs.flat:
        ax.set_xlabel(r'Raman shift (cm$^{-1}$)')
        ax.legend(fontsize=8)
    return fig


def plot_hidden_size(ax, sub):
    """Final S10 statistic: median across windows for the single base seed 42."""
    sub = sub.sort_values('hidden_size')
    ax.plot(sub.hidden_size, sub.median_q, color=RED, label='Median q; base seed 42')
    baseline = sub[sub.hidden_size == 64]
    if len(baseline) != 1:
        raise ValueError('Expected one H=64 reference per dataset')
    ax.axvline(64, color='0.5', ls=':', lw=1)
    ax.scatter([64], baseline.median_q, color='black', zorder=4, label='Primary H = 64')
    ax.set(xlabel='Hidden size H', ylabel='Median recovery score q')
    ax.legend(fontsize=8)
