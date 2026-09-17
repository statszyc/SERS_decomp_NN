"""Replot final numerical panels from the released data, without retraining.

The original publication layouts are archived separately in figures/. These
compact plots reproduce the numerical content, not pixel-identical typography.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io import ROOT
from .selection import consensus
from .figure_panels import (
    plot_figure1, plot_figure3, plot_consensus_entry, plot_s3_overview,
    diagnostic_figure, plot_hidden_size,
)

RED='#D92323';GREEN='#007A3D';BLUE='#1558A6';GOLD='#D4AC0D'
COLORS={'reference':'black','Reference':'black','Ref.':'black','Selected':RED,
        'selected':RED,'Loss comp.':GREEN,'Loss Comp.':GREEN,'loss':GREEN,'Oracle':GOLD,
        'Loss comparator':GREEN,'theta*_Pyo':RED,r'$\theta^*_{\mathrm{Pyo}}$':RED,
        r'$\theta^{*}_{\mathrm{Pyo}}$':RED,
        'Rec-loss selected':GREEN,'Stability-selected':RED}
LABELS={'Rec-loss selected':'Loss Comp.','Stability-selected':'Selected',
        'Reference':'Ref.','Loss comp.':'Loss Comp.'}

def frame(name):return pd.read_csv(ROOT/'data/figures'/name)

def finish(fig,name,out):
    if not fig.get_constrained_layout():
        # The overview reserves a top strip for its shared legend.
        fig.tight_layout(rect=(0,0,1,.97) if name=='Figure_S3_overview' else None)
    for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

def spectra(ax,df,x,y,group):
    for key,part in df.groupby(group,sort=False):
        ax.plot(part[x],part[y],label=LABELS.get(key,key),color=COLORS.get(key),lw=1.2)
    ax.set_xlabel(r'Raman shift (cm$^{-1}$)');ax.set_ylabel('Display intensity (a.u.)')
    ax.legend(fontsize=8)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('reproduced/plots'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    finish(plot_figure1(),'Figure_1_B_E',a.output)
    qz=np.load(ROOT/'results/validation_scores.npz');q=qz['q'];ids=qz['theta_ids'];tids=qz['task_ids']
    ranks,entry,order=consensus(q,ids)
    finish(plot_figure3(),'Figure_3_A_E',a.output)
    val=pd.read_csv(ROOT/'results/validation_selected_vs_loss.csv');index=pd.read_json(ROOT/'data/validation_index.json')
    val['dataset']=val.task_id.map(index.set_index('task_id').dataset_id)
    val['oracle_q']=val.task_id.map(dict(zip(tids,q.max(1))))
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    plot_consensus_entry(axs[0,0],entry)
    groups=list(val.dataset.unique());xs=np.arange(len(groups))
    for j,(col,label,color) in enumerate([('loss_q','Loss Comp.',GREEN),('selected_q','Selected',RED),('oracle_q','Oracle',GOLD)]):
        sub=val.groupby('dataset')[col].agg(['mean','std']).loc[groups]
        axs[0,1].bar(xs+(j-1)*.24,sub['mean'],.23,yerr=sub['std'],label=label,color=color,capsize=2)
    axs[0,1].set_xticks(xs,[s.replace('_full','').replace('_','\n') for s in groups],fontsize=8);axs[0,1].set(ylabel='Recovery score q',title='Figure 4B');axs[0,1].legend(fontsize=8)
    data=frame('figure4_spectra.csv')
    representative_tasks=['dnarna_experimental_task_05_07__perturb_fold_00',
                          'pyocyanin_experimental_task_01_03__perturb_fold_00']
    for panel,ax,task in zip(['(C)','(D)'],axs[1],representative_tasks):
        spectra(ax,data[data.panel==panel],'wavenumber','normalized_intensity','curve')
        record=val.set_index('task_id').loc[task]
        ax.set_title(f'Figure 4{panel}: q Selected={record.selected_q:.5f}; Loss Comp.={record.loss_q:.5f}',fontsize=10)
    finish(fig,'Figure_4',a.output)
    fig=plt.figure(figsize=(11,7));gs=fig.add_gridspec(2,2)
    upper=[fig.add_subplot(gs[0,j]) for j in range(2)];lower=fig.add_subplot(gs[1,:])
    df=frame('figure5_spectra.csv')
    for panel,ax in zip('AB',upper):
        sub=df[df.panel==panel];spectra(ax,sub,'raman_shift_cm_1','display_intensity','curve');ax.set_title('Ad5 '+str(sub.task_window.iloc[0]))
    scores=pd.read_csv(ROOT/'results/test_scores.csv');datasets=list(scores.dataset.unique())
    delta=scores.global_q-scores.loss_q
    np.testing.assert_allclose(delta,scores.delta_q_global_minus_loss,rtol=0,atol=1e-14)
    parts=[delta[scores.dataset==d].to_numpy() for d in datasets]
    bp=lower.boxplot(parts,tick_labels=datasets,whis=1.5,showfliers=False,patch_artist=True)
    for box in bp['boxes']:box.set_facecolor(RED);box.set_alpha(.3)
    lower.axhline(0,color='black',lw=.8)
    lower.set(title='Figure 5C: test recovery difference',xlabel='Virus type',ylabel=r'$\Delta q_i$ (selected minus loss)')
    finish(fig,'Figure_5',a.output)
    finish(plot_s3_overview(),'Figure_S3_overview',a.output)
    fig,axs=plt.subplots(2,1,figsize=(8,6));df=frame('figureS3_recovery.csv')
    for panel,ax in zip(['C_before','C_after'],axs):spectra(ax,df[df.panel==panel],'wavenumber','plotted_intensity','curve');ax.set_title(panel.replace('_',' '))
    finish(fig,'Figure_S3_C',a.output)
    df=frame('figureS4_scores.csv');groups=list(df.dataset_id.unique());fig,axs=plt.subplots(2,2,figsize=(12,8))
    for dataset,ax in zip(groups,axs.flat):
        sub=df[df.dataset_id==dataset];methods=list(sub.method.unique())
        ax.boxplot([sub[sub.method==m].q for m in methods],tick_labels=methods,showmeans=True)
        ax.tick_params(axis='x',labelrotation=25);ax.set_title(dataset);ax.set_ylabel('q')
    finish(fig,'Figure_S4_scores',a.output)
    df=frame('figureS4_curves.csv');groups=list(df.task_id.unique());fig,axs=plt.subplots(1,len(groups),figsize=(6*len(groups),4),squeeze=False)
    for task,ax in zip(groups,axs.flat):spectra(ax,df[df.task_id==task],'raman_shift_cm-1','display_intensity','method');ax.set_title(task.split('__')[0],fontsize=9)
    finish(fig,'Figure_S4_curves',a.output)
    for num,filename,method,zoom in [(5,'figureS5_data.csv','prior_dual_network',(1570,1680)),
                                    (9,'figureS9.csv','mcrals',(980,1100))]:
        finish(diagnostic_figure(frame(filename),num,method,zoom),f'Figure_S{num}',a.output)
    for num in [6,7,8,10]:
        df=frame(f'figureS{num}_data.csv');datasets=list(df.dataset_id.unique());fig,axs=plt.subplots((len(datasets)+1)//2,2,figsize=(12,3.1*((len(datasets)+1)//2)),squeeze=False)
        for dataset,ax in zip(datasets,axs.flat):
            sub=df[df.dataset_id==dataset]
            if num==6:
                sizes=sorted(sub.window_size.unique());ax.boxplot([sub[sub.window_size==k].q for k in sizes],tick_labels=sizes,showmeans=True);ax.set(xlabel='Window size K',ylabel='q')
            elif num==7:ax.plot(sub.display_position,sub.q,'-o',ms=3);ax.set(xlabel='Window position',ylabel='q')
            elif num==8:
                for key,part in sub.groupby('coefficient_family'):ax.plot(part.display_position,part.coefficient_value,'-o',ms=3,label=key)
                ax.set_xlabel('Window position');ax.legend(fontsize=8)
            else:
                plot_hidden_size(ax,sub)
            ax.set_title(dataset,fontsize=9)
        for ax in list(axs.flat)[len(datasets):]:ax.set_visible(False)
        finish(fig,f'Figure_S{num}',a.output)
    print('Saved numerical replots to',a.output)

if __name__=='__main__':main()
