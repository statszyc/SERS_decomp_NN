import unittest
import numpy as np
import torch
from sers.io import tasks,grid,task_seed,training_input,sliding_windows
from sers.selection import consensus,loss_comparators
from sers.training import run_single_trial
from sers.baselines import fit_fixed_bg_mcr

class ContractTests(unittest.TestCase):
    def test_reference_boundary(self):
        t=tasks('test')[0];clean=training_input(t)
        self.assertIsNone(clean['gt_f']);self.assertIsNotNone(t['gt_f'])
        self.assertEqual(clean['x_mix'],t['x_mix'])

    def test_grid(self):
        rows=grid();self.assertEqual(len(rows),7560)
        self.assertEqual(len({tuple(sorted(r['theta'].items())) for r in rows}),7560)
        self.assertEqual({r['theta']['hidden_dim'] for r in rows},{64})
        self.assertEqual(task_seed(0),2391430712)

    def test_ranking_and_loss(self):
        ids=np.array(['b','a','c']);q=np.array([[.8,.8,.1],[.7,.9,.2]])
        ranks,entry,order=consensus(q,ids)
        np.testing.assert_array_equal(ranks,[[2,1,3],[2,1,3]])
        self.assertEqual(ids[order[0]],'a')
        np.testing.assert_array_equal(loss_comparators([[.1,.1,.2]],ids),[1])

    def test_windows(self):
        result=list(sliding_windows([0,1],np.ones((8,2)),range(8),[[1,1]],3))
        self.assertEqual(len(result),6);self.assertEqual(result[-1]['concentrations'],[5,6,7])

    def test_synthetic_execution(self):
        # Tiny non-scientific smoke test, not a rerun of the paper's training.
        torch.set_num_threads(1)
        x=np.linspace(400,1700,24);bg=np.ones(24);signal=np.exp(-((x-1000)/100)**2)
        t=dict(x_axis=x.tolist(),x_mix=np.array([bg+c*signal for c in [.1,.5,.9]]).tolist(),
               concentrations=[0,1,2],bg=[bg.tolist()],gt_f=None,metadata={},dataset_id='synthetic',task_id='smoke')
        theta=dict(num_frequencies=2,scale_factor=1,hidden_dim=8,z_dim=8,lambda_penalty=1,lambda_flat=.0001,lr=.001)
        first=run_single_trial(t,theta,'smoke',epochs=2,device='cpu',torch_seed=123,return_artifacts=True)
        second=run_single_trial(t,theta,'smoke',epochs=2,device='cpu',torch_seed=123,return_artifacts=True)
        np.testing.assert_array_equal(first['artifacts']['f_final'],second['artifacts']['f_final'])
        self.assertTrue(np.isfinite(first['output_features']['residual_norm_ratio']))
        self.assertIsNone(first['ground_truth_metrics']['gt_cosine'])
        conventional=fit_fixed_bg_mcr(t,max_iter=5)
        self.assertTrue(np.isfinite(conventional['curve']).all())

if __name__=='__main__':unittest.main()
