"""Semantic regression checks, independent of pixel-identical typography."""
import hashlib
import json
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sers.io import ROOT
from sers.figure_panels import (
    unit_area, figure1_display, pyocyanin_selection, plot_figure3,
    plot_s3_overview, s3_backgrounds, diagnostic_figure, plot_hidden_size,
    plot_consensus_entry, VALIDATION_DATASETS, TEST_DATASETS,
)
from sers.plot_results import spectra, GREEN, RED
from sers.selection import consensus


class PlottingTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_figure1_signed_trapezoidal_normalization(self):
        x = np.array([0., 1., 3.])
        np.testing.assert_allclose(unit_area(x, [1, 2, 4]), np.array([1, 2, 4])/7.5)
        np.testing.assert_allclose(unit_area([0, 1, 2], [-1, 0, 1]), [-1, 0, 1])
        for panel in 'BCDE':
            with np.load(ROOT/f'data/figures/figure1_{panel}.npz') as payload:
                display = figure1_display(payload)
                reference = payload['reference_curve'].astype(float)
                x = payload['x_axis'].astype(float)
                ref_unit = reference/np.trapezoid(reference, x)
                peak = np.max(np.abs(ref_unit))
                for key, actual in display.items():
                    y = payload[key].astype(float)
                    expected = y/np.trapezoid(y, x)/peak
                    np.testing.assert_allclose(actual, expected, rtol=1e-13)

    def test_figure3_data_driven_panel_a(self):
        data = pyocyanin_selection()
        selected = data['order'][0]
        self.assertEqual(data['ids'][selected], 'h64revised_theta_02018')
        np.testing.assert_array_equal(data['ranks'][:, selected], [210,519,860,1030,957,1042])
        self.assertAlmostEqual(data['entry'][selected], 1042/7560)
        self.assertLess(957/7560, .2)
        self.assertGreater(957/7560, .1)
        fig = plot_figure3()
        self.assertEqual(len(fig.axes), 7)  # three parts of A plus B–E
        self.assertEqual(len(fig.axes[0].lines), 4)  # triplet and background
        np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(),
                                   data['ranks'][:, selected]/7560)
        self.assertEqual([t.get_text() for t in fig.axes[-1].get_legend().texts],
                         ['Ref.', 'Loss Comp.', 'Selected'])

    def test_s3_backgrounds_and_order(self):
        source = s3_backgrounds()
        self.assertEqual(set(source.dataset_id), set(VALIDATION_DATASETS+TEST_DATASETS))
        for _, part in source.groupby('dataset_id'):
            y = part.source_intensity.to_numpy()
            expected = y/np.percentile(np.abs(y), 95)-.9
            np.testing.assert_allclose(part.display_intensity, expected, atol=2e-14)
        provenance = json.loads((ROOT/'metadata/plot_additions.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/provenance['output']['file']).read_bytes()).hexdigest(),
                         provenance['output']['sha256'])
        fig = plot_s3_overview()
        for ax in fig.axes:
            self.assertEqual(len(ax.lines), 6)
            self.assertEqual(ax.lines[-1].get_label(), 'Background')
            self.assertEqual(ax.lines[-1].get_linestyle(), '--')
        self.assertEqual(fig.axes[0].get_title(), 'Pyocyanin experimental')
        self.assertEqual(fig.axes[1].get_title(), 'Pyocyanin simulated')

    def test_diagnostic_zoom_panels(self):
        for n, filename, method, bounds, flag in [
            (5, 'figureS5_data.csv', 'prior_dual_network', (1570,1680), 'in_zoom_1570_1680'),
            (9, 'figureS9.csv', 'mcrals', (980,1100), 'in_peak_region_980_1100')]:
            df = pd.read_csv(ROOT/'data/figures'/filename)
            np.testing.assert_array_equal(df['raman_shift_cm-1'].between(*bounds), df[flag])
            fig = diagnostic_figure(df, n, method, bounds)
            self.assertEqual(len(fig.axes), 4)
            self.assertEqual(fig.axes[1].get_xlim(), bounds)
            self.assertEqual(fig.axes[3].get_xlim(), bounds)
            for ax in fig.axes:
                self.assertTrue(all(line.get_linestyle() == '-' for line in ax.lines))
            for key in ['selected', method]:
                np.testing.assert_allclose(df[key+'_absolute_deviation'],
                                           np.abs(df[key]-df.reference), atol=1e-14)

    def test_s10_median_single_seed_not_mean(self):
        source = pd.read_csv(ROOT/'data/figures/figureS10_data.csv')
        results = pd.read_csv(ROOT/'results/hidden_size/h_sensitivity_task_results.csv')
        median = (results[results.base_seed == 42]
                  .groupby(['dataset_id','hidden_size']).gt_cosine.median())
        comparison = source.set_index(['dataset_id','hidden_size']).sort_index()
        np.testing.assert_allclose(comparison.median_q, median.loc[comparison.index], atol=1e-14)
        sub = source[source.dataset_id == source.dataset_id.iloc[0]].sort_values('hidden_size')
        _, ax = plt.subplots()
        plot_hidden_size(ax, sub)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), sub.median_q)
        self.assertFalse(np.allclose(sub.median_q, sub.mean_q))
        self.assertEqual(len(ax.collections), 1)  # H64 marker, no uncertainty band

    def test_figure4_semantics(self):
        with np.load(ROOT/'results/validation_scores.npz') as payload:
            _, entry, _ = consensus(payload['q'], payload['theta_ids'])
        _, ax = plt.subplots()
        plot_consensus_entry(ax, entry)
        self.assertTrue(ax.get_xlim()[0] < 2882/7560 < ax.get_xlim()[1])
        self.assertLess(ax.get_ylim()[1], 7560)
        df = pd.read_csv(ROOT/'data/figures/figure4_spectra.csv')
        _, ax = plt.subplots()
        spectra(ax, df[df.panel == '(C)'], 'wavenumber','normalized_intensity','curve')
        mapped = {line.get_label(): line.get_color() for line in ax.lines}
        self.assertEqual(mapped['Selected'], RED)
        self.assertEqual(mapped['Loss Comp.'], GREEN)


if __name__ == '__main__':
    unittest.main()
