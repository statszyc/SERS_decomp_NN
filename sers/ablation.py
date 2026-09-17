"""Replay one published Figure 1 matched ablation case."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .io import ROOT,tasks,training_input
from .training import run_single_trial
from .features import compute_ground_truth_metrics
from .replay import plain

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',choices=list('BCDE'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Output exists; choose a new path')
    row=pd.read_csv(ROOT/'data/figures/figure1_cases.csv').set_index('panel').loc[a.panel]
    task=next(t for t in tasks('validation') if t['task_id']==row.task_id)
    theta=json.loads(row.theta_json);seed=int(row.torch_seed)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    outputs={}
    for mode in ['retained','ablated']:
        trial=dict(theta);disable=False
        if mode=='ablated':
            if row.component=='fourier_zero':disable=True
            elif row.component=='penalty_zero':trial['lambda_penalty']=0.0
            elif row.component=='flatness_zero':trial['lambda_flat']=0.0
            elif row.component=='scale_one':trial['scale_factor']=1.0
            else:raise ValueError(row.component)
        result=run_single_trial(training_input(task),trial,f'{a.panel}_{mode}',epochs=1000,device='cpu',torch_seed=seed,return_artifacts=True,disable_fourier=disable)
        ref=np.asarray(task['gt_f'],np.float32)*trial['scale_factor']
        result['postfit_reference_metrics']=compute_ground_truth_metrics(ref,np.asarray(result['artifacts']['f_final'],np.float32))
        outputs[mode]=plain(result)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(outputs,indent=2)+'\n');print(a.output)

if __name__=='__main__':main()
