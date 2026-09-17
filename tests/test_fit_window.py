import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sers.fit_window import demo_frame, parse_window, fit_window, save_outputs, SELECTED_ID


class UserWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.frame = demo_frame()
        cls.result = fit_window(cls.frame, [0,1,2], epochs=2)

    def test_reference_cannot_affect_training(self):
        frame = self.frame.drop(columns='reference')
        other = fit_window(frame, [0,1,2], epochs=2)
        np.testing.assert_array_equal(self.result['artifacts']['f_final'],
                                      other['artifacts']['f_final'])
        self.assertIsNone(other['postfit_reference_metrics']['gt_cosine'])
        self.assertIsNotNone(self.result['postfit_reference_metrics']['gt_cosine'])
        self.assertEqual(self.result['trial_id'], SELECTED_ID)
        task, reference = parse_window(self.frame, [0,1,2])
        self.assertIsNone(task['gt_f'])
        self.assertIsNotNone(reference)

    def test_bad_inputs_rejected(self):
        for frame, levels in [
            (self.frame.iloc[::-1], [0,1,2]),
            (self.frame, [0,1]),
            (self.frame, [0,0,2]),
            (self.frame.rename(columns={'mix_1':'mix_7'}), [0,1,2]),
            (self.frame.assign(background=np.nan), [0,1,2]),
            (self.frame.assign(background=0), [0,1,2]),
        ]:
            with self.assertRaises(ValueError):
                parse_window(frame, levels)

    def test_outputs_and_factorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'fit'
            save_outputs(self.frame, [0,1,2], self.result, out)
            curves = pd.read_csv(out/'curves.csv')
            coefficients = pd.read_csv(out/'coefficients.csv')
            for j, row in coefficients.iterrows():
                rebuilt = (row.analyte_coefficient*curves.recovered_analyte +
                           row.background_coefficient*self.frame.background)
                np.testing.assert_allclose(rebuilt, curves[f'reconstructed_mix_{j}'],
                                           rtol=2e-6, atol=2e-7)
            self.assertEqual({p.name for p in out.iterdir()}, {
                'input.csv','fit.json','curves.csv','coefficients.csv','recovery.png','recovery.pdf'})
            with self.assertRaises(FileExistsError):
                save_outputs(self.frame, [0,1,2], self.result, out)

    def test_degenerate_background_area_rejected(self):
        frame = pd.DataFrame({'wavenumber': np.arange(450,458),
                              'background': [-1,1,-1,1,-1,1,-1,1],
                              'mix_0': np.arange(1,9), 'mix_1': np.arange(2,10)})
        with self.assertRaisesRegex(ValueError, 'signed background area'):
            fit_window(frame, [0,1], epochs=2)

    def test_float32_coordinate_collapse_rejected(self):
        with self.assertRaisesRegex(ValueError, 'distinct and increasing in float32'):
            fit_window(self.frame, [1e8,1e8+1,1e8+2], epochs=2)
        frame = self.frame.assign(wavenumber=1e10+np.arange(len(self.frame)))
        with self.assertRaisesRegex(ValueError, 'distinct and increasing in float32'):
            fit_window(frame, [0,1,2], epochs=2)

    def test_float32_overflow_rejected(self):
        frame = self.frame.assign(background=1e39)
        with self.assertRaisesRegex(ValueError, 'remain finite'):
            fit_window(frame, [0,1,2], epochs=2)


if __name__ == '__main__':
    unittest.main()
