"""Run final models on frozen inputs; no search over random seeds.

Examples:
  python -m sers.replay --split validation --task 0
  python -m sers.replay --split test --task 0 --candidate loss
  python -m sers.replay --analysis robustness --split test --task 0 --base-seed 18
  python -m sers.replay --analysis hidden --task 0 --hidden-size 32 --base-seed 18
  python -m sers.replay --analysis window --task 0
  python -m sers.replay --method mcrals --task 0
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .io import ROOT, grid, tasks, task_seed, training_input, load_json
from .training import run_single_trial
from .features import compute_ground_truth_metrics
from .baselines import fit_fixed_bg_mcr, fit_prior_nn

def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k,v in value.items()}
    if isinstance(value, list): return [plain(v) for v in value]
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--task',type=int,required=True)
    p.add_argument('--candidate',default='selected',help='selected, pyo, loss, or exact theta_id')
    p.add_argument('--method',choices=['neural','mcrals','prior'],default='neural')
    p.add_argument('--analysis',choices=['primary','robustness','hidden','window'],default='primary')
    p.add_argument('--base-seed',type=int,choices=[18,42,251],default=18)
    p.add_argument('--hidden-size',type=int,choices=range(16,129),default=64)
    p.add_argument('--grid-start',type=int)
    p.add_argument('--grid-stop',type=int,help='Exclusive; explicit chunk of the final 7560 grid')
    p.add_argument('--output',type=Path,default=Path('reproduced/replay.json'))
    a=p.parse_args()
    if a.output.exists(): p.error('Output exists; choose a new path')
    if a.analysis=='hidden' and a.split!='validation': p.error('H sensitivity is validation-only')
    if a.analysis!='primary' and (a.candidate!='selected' or a.method!='neural' or a.grid_start is not None):
        p.error('Sensitivity analyses use only the frozen selected neural configuration')
    split='window_size' if a.analysis=='window' else a.split
    collection=tasks(split)
    if not 0<=a.task<len(collection): p.error('Task index out of range')
    task=collection[a.task]
    if a.analysis=='window': seed=load_json('data/window_size_plan.json')[a.task]['torch_seed']
    elif a.analysis in ['robustness','hidden']: seed=task_seed(a.task,a.base_seed,20260810)
    else: seed=task_seed(a.task)
    rows=grid(); byid={r['theta_id']:r for r in rows}
    name={'selected':'h64revised_theta_01308','pyo':'h64revised_theta_02018'}.get(a.candidate,a.candidate)
    if a.candidate=='loss':
        filename='test_loss_selection.csv' if a.split=='test' else 'validation_selected_vs_loss.csv'
        df=pd.read_csv(ROOT/'results'/filename)
        name=df.set_index('task_id').loc[task['task_id'],'loss_theta_id']
    if a.grid_start is not None or a.grid_stop is not None:
        if a.grid_start is None or a.grid_stop is None or not 0<=a.grid_start<a.grid_stop<=len(rows): p.error('Specify a valid explicit grid range')
        if a.method!='neural': p.error('Grid evaluation applies to neural decomposition')
        chosen=rows[a.grid_start:a.grid_stop]
    else:
        if name not in byid: p.error('Unknown candidate ID')
        chosen=[byid[name]]
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    out=[]
    for row in chosen:
        theta=dict(row['theta'])
        if a.analysis=='hidden': theta.update(hidden_dim=a.hidden_size,z_dim=a.hidden_size)
        fit_task=training_input(task)
        if a.method=='neural':
            result=run_single_trial(fit_task,theta,row['theta_id'],epochs=1000,device='cpu',torch_seed=seed,return_artifacts=True)
            reference=np.asarray(task['gt_f'],dtype=np.float32)*theta['scale_factor']
            curve=np.asarray(result['artifacts']['f_final'],dtype=np.float32)
        else:
            result=fit_fixed_bg_mcr(fit_task) if a.method=='mcrals' else fit_prior_nn(fit_task,epochs=1000,random_seed=seed)
            reference=np.asarray(task['gt_f'],dtype=np.float32)
            curve=np.asarray(result['curve'],dtype=np.float32)
        result['postfit_reference_metrics']=compute_ground_truth_metrics(reference,curve)
        out.append({'task_id':task['task_id'],'theta_id':row['theta_id'],'torch_seed':seed,'method':a.method,'analysis':a.analysis,'result':plain(result)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(a.output)

if __name__=='__main__': main()
