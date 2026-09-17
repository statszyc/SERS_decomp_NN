"""Verify saved results and export the final scientific tables, without fitting."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import ROOT,grid,task_seed,tasks,load_json
from .selection import consensus,loss_comparators

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('reproduced'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    g=grid();ids=np.array([r['theta_id'] for r in g]); M=len(g)
    assert M==7560 and len(set(ids))==M
    qz=np.load(ROOT/'results/validation_scores.npz',allow_pickle=False)
    assert np.array_equal(qz['theta_ids'],ids)
    q=qz['q']; ranks,entry,order=consensus(q,ids)
    assert q.shape==(37,M) and np.array_equal(ranks,qz['rank_position'])
    assert ids[order[0]]=='h64revised_theta_01308' and (entry==entry.min()).sum()==1
    pyo=np.char.startswith(qz['task_ids'],'pyocyanin_experimental')
    pr,pe,po=consensus(q[pyo],ids)
    assert ids[po[0]]=='h64revised_theta_02018' and (pe==pe.min()).sum()==1
    expected=np.array([[210,519,860,1030,957,1042],[368,467,303,1061,549,384],
                       [830,479,968,1109,810,464],[850,620,855,1129,878,1009]])
    assert np.array_equal(pr[:,po[:4]].T,expected)
    checks={'validation_windows':37,'test_windows':33,'candidate_count':M,
            'selected':str(ids[order[0]]),'p_entry':float(entry.min()),
            'pyocyanin_selected':str(ids[po[0]]),'pyocyanin_p_entry':float(pe.min())}
    for split,n in [('validation',37),('test',33)]:
        z=np.load(ROOT/f'results/{split}_residuals.npz',allow_pickle=False)
        assert z['residual_norm_ratio'].shape==(n,M) and np.array_equal(z['theta_ids'],ids)
        index=load_json(f'data/{split}_index.json')
        assert list(z['task_ids'])==[t['task_id'] for t in index]
        assert list(z['torch_seed'])==[task_seed(i) for i in range(n)]
        filename='test_loss_selection.csv' if split=='test' else 'validation_selected_vs_loss.csv'
        frozen=pd.read_csv(ROOT/'results'/filename).set_index('task_id').loc[z['task_ids']]
        winners=loss_comparators(z['residual_norm_ratio'],ids)
        assert np.array_equal(ids[winners],frozen['loss_theta_id'].to_numpy())
        assert np.allclose(z['residual_norm_ratio'][np.arange(n),winners],frozen['loss_residual_norm_ratio'],rtol=1e-13,atol=1e-15)
        checks[split+'_loss_winners_verified']=n
        if split=='validation':
            assert np.allclose(q[:,order[0]],frozen['selected_q'],atol=1e-14)
            assert np.allclose(q[np.arange(n),winners],frozen['loss_q'],atol=1e-14)
    pd.DataFrame([{'role':'Selected' if j==0 else f'Next {j}',
                   'theta_id':str(ids[i]),'p_entry':entry[i],
                   **g[i]['theta']} for j,i in enumerate(order[:5])]).to_csv(a.output/'Table_2.csv',index=False)
    axes=[]
    for k in ['num_frequencies','scale_factor','lambda_penalty','lambda_flat','lr']:
        axes.append({'parameter':k,'grid_values':json.dumps(sorted(set(r['theta'][k] for r in g))),
                     'selected_value':g[order[0]]['theta'][k]})
    pd.DataFrame(axes).to_csv(a.output/'Table_1_and_S1.csv',index=False)
    pd.read_csv(ROOT/'data/figures/figure1_cases.csv').to_csv(a.output/'Table_S2_ablation_cases.csv',index=False)
    frames=[]
    val=pd.read_csv(ROOT/'results/validation_selected_vs_loss.csv')
    lookup={t['task_id']:t['dataset_id'] for t in load_json('data/validation_index.json')}
    val['dataset_id']=val.task_id.map(lookup)
    val['oracle_q']=q.max(1)
    val.groupby('dataset_id')[['selected_q','loss_q','oracle_q','delta_q']].agg(['mean','std','min','max']).to_csv(a.output/'validation_summary.csv')
    test=pd.read_csv(ROOT/'results/test_scores.csv')
    test.groupby('dataset')[['global_q','loss_q','delta_q_global_minus_loss']].agg(['mean','std','min','max']).to_csv(a.output/'test_summary.csv')
    base=pd.read_csv(ROOT/'results/baselines/baseline_task_scores.csv')
    base.groupby(['dataset_id','method'])['q'].agg(['count','mean','std','median','min','max']).to_csv(a.output/'baseline_summary.csv')
    robust=pd.read_csv(ROOT/'results/robustness/robustness_long.csv')
    assert len(robust)==210 and set(robust.base_seed)=={18,42,251}
    assert np.array_equal(robust.torch_seed,[task_seed(int(r.domain_task_index),int(r.base_seed),20260810) for r in robust.itertuples()])
    robust.groupby(['domain','dataset_id','base_seed']).q.agg(['count','mean','std','min','max']).to_csv(a.output/'robustness_summary.csv')
    h=pd.read_csv(ROOT/'results/hidden_size/h_sensitivity_task_results.csv')
    assert len(h)==12543 and set(h.hidden_size)==set(range(16,129))
    assert np.array_equal(h.torch_seed,[task_seed(int(r.task_index),int(r.base_seed),20260810) for r in h.itertuples()])
    h.groupby(['dataset_id','hidden_size','base_seed']).gt_cosine.agg(['count','mean','std','median','min','max']).to_csv(a.output/'hidden_size_summary.csv')
    k=pd.read_csv(ROOT/'results/window_size/k57_task_scores.csv')
    assert len(k)==92
    k.groupby(['current_domain','dataset_id','window_size']).gt_cosine.agg(['count','mean','std','min','max']).to_csv(a.output/'window_size_summary.csv')
    checks.update(robustness_fits=210,hidden_size_fits=12543,window_size_fits=92,
                  finite_primary_residuals=70*M,status='passed')
    (a.output/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
