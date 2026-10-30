"""Consolidate recorded learning curves and derive final test NLL from cached probabilities.

Runs no model inference or training. Source arrays and reported accuracies are checked.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG = Path(__file__).resolve().parent
ROOT = FIG.parents[2]
OUT = FIG / 'supervisor_comparison'
POLICIES = ['baseline', 'rot30', 'rot45', 'rot60', 'rot100', 'rot150']
LABELS = {'baseline': 'Original baseline', **{f'rot{a}': f'±{a}°' for a in (30,45,60,100,150)}}
COLORS = dict(zip(POLICIES, ['#666666','#0072B2','#009E73','#D55E00','#AA4499','#882255']))
STYLES = dict(zip(POLICIES, [':','-','--','-.',(0,(5,2,1,2)),(0,(2,2))]))
MODES = [('none','Single image','o'),('mean','Seven-angle mean','s'),('max','Seven-angle max','^')]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read_csv(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))
def normalize_path(p): return p.split('/macaque_crops/',1)[1]

def main():
    OUT.mkdir(exist_ok=True)
    original = json.loads((FIG/'plot_data.json').read_text())
    follow = json.loads((FIG/'followup_loss_data.json').read_text())
    curves={}; sources={}; best={**original['best_epochs'],**follow['best_epochs']}
    for key in POLICIES:
        source=(original if key in original['curve_sources'] else follow)['curve_sources'][key]
        path=ROOT/source['path']
        assert sha(path)==source['sha256'],path
        rows=[{k:int(v) if k=='epoch' else float(v) for k,v in r.items()} for r in read_csv(path)]
        assert [r['epoch'] for r in rows]==list(range(1,31))
        assert rows[best[key]-1]['val_top1']==max(r['val_top1'] for r in rows)
        curves[key]=rows; sources[key]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}

    base_path=next((ROOT/'artifacts/final_eval').glob('*aug3_canonical_20261014_metrics_summary.json'))
    base=json.loads(base_path.read_text())
    metrics={'baseline':{'none':{'top1':base['top1'],'f1_macro':base['f1_macro'],
                                 'nll':None,'nll_reason':'Full baseline probabilities were not saved.',
                                 'correct':round(1498*base['top1'])}}}
    evidence={'baseline_summary':{'path':str(base_path.relative_to(ROOT)),'sha256':sha(base_path)}}
    reference_pairs=None
    for key in POLICIES[1:]:
        angle=int(key[3:])
        suite=ROOT/'artifacts/augmentation_sweeps'/('20261028T164412Z_JmVXbV' if angle in (45,60) else '20261025T023752Z_jyY1BT')
        eval_dir=suite/'eval'
        summaries={}
        for p in eval_dir.glob('*_metrics_summary.json'):
            if f'_rot{angle}_' not in p.name: continue
            d=json.loads(p.read_text())
            if d['split']=='test': summaries[d['tta_mode']]=(p,d)
        assert set(summaries)=={'none','mean','max'}
        metrics[key]={}; evidence[key]={}
        mean_path=next(eval_dir.glob(f'*rot{angle}*mean_test*view_probabilities.npz'))
        with np.load(mean_path,allow_pickle=False) as z:
            assert z['angles'].tolist()==[-30,-20,-10,0,10,20,30]
            labels=z['labels'].copy(); paths=z['paths'].copy(); classes=z['class_names'].copy()
            probs=z['probabilities'][:,3,:].copy()
        assert probs.shape==(1498,149)
        pairs=[(normalize_path(p),str(classes[t])) for p,t in zip(paths,labels)]
        assert len(set(pairs))==1498
        if reference_pairs is None: reference_pairs=set(pairs)
        assert set(pairs)==reference_pairs
        for mode,_,_ in MODES:
            p,summary=summaries[mode]
            array_path=mean_path if mode=='none' else next(eval_dir.glob(f'*rot{angle}*{mode}_test*view_probabilities.npz'))
            if mode!='none':
                with np.load(array_path,allow_pickle=False) as z:
                    assert np.array_equal(z['labels'],labels)
                    assert np.array_equal(z['paths'],paths)
                    assert np.array_equal(z['class_names'],classes)
                    probs=z['pooled_probabilities'].copy()
            assert np.allclose(probs.sum(axis=1),1,atol=2e-6)
            true_probs=probs[np.arange(len(labels)),labels].astype(np.float64)
            assert np.all(true_probs>0) and np.all(true_probs<=1)
            predictions=probs.argmax(axis=1)
            accuracy=float((predictions==labels).mean())
            assert abs(accuracy-summary['top1'])<1e-12
            hard_path=p.with_name(p.name.replace('_metrics_summary.json','_hard_examples.csv'))
            hard={(normalize_path(r['image_path']),r['true_id']):r for r in read_csv(hard_path)}
            assert set(hard)==set(pairs)
            for i,pair in enumerate(pairs):
                assert str(classes[predictions[i]])==hard[pair]['pred_id']
                assert abs(float(probs[i].max())-float(hard[pair]['top1_prob']))<2e-5
            metrics[key][mode]={'top1':accuracy,'f1_macro':summary['f1_macro'],
                                'nll':float(-np.log(true_probs).mean()),
                                'correct':int((predictions==labels).sum()),
                                'min_true_class_probability':float(true_probs.min())}
            evidence[key][mode]={'summary':str(p.relative_to(ROOT)),'summary_sha256':sha(p),
                                  'arrays':str(array_path.relative_to(ROOT)),'arrays_sha256':sha(array_path),
                                  'single_view_source':'0° view from mean-TTA cache; per-image predictions/confidences verified against recorded single-image output' if mode=='none' else None}
    baseline_hard=base_path.with_name(base_path.name.replace('_metrics_summary.json','_hard_examples.csv'))
    baseline_pairs={(normalize_path(r['image_path']),r['true_id']) for r in read_csv(baseline_hard)}
    assert baseline_pairs==reference_pairs

    plt.rcParams.update({'font.family':'Carlito','font.size':10,'axes.titlesize':12,
                         'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':9})
    legend=[Line2D([0],[0],color=COLORS[k],ls=STYLES[k],lw=2,label=LABELS[k]) for k in POLICIES]
    legend_rows=[legend[i] for i in (0,3,1,4,2,5)]
    def save(fig,name):
        for ext in ('png','pdf'): fig.savefig(OUT/f'{name}.{ext}',dpi=260)
        plt.close(fig)

    fig,axs=plt.subplots(2,1,figsize=(6.5,5.8),layout='constrained')
    fig.legend(handles=legend_rows,loc='outside upper center',ncol=3,frameon=False)
    for ax,field,title in zip(axs,['train_loss','val_loss'],['A  Training loss · augmented training inputs','B  Validation loss · common unaugmented split']):
        for key in POLICIES:
            ax.plot(range(1,31),[r[field] for r in curves[key]],color=COLORS[key],ls=STYLES[key],lw=1.8)
        ax.set(title=title,xlabel='Epoch',ylabel='Margin-adjusted cross-entropy',xlim=(1,30),ylim=(0,20))
        ax.set_xticks([1,5,10,15,20,25,30]);ax.set_yticks([0,5,10,15,20]);ax.grid(alpha=.18)
    save(fig,'all_policy_loss')

    fig,axs=plt.subplots(2,1,figsize=(6.5,6.9),layout='constrained')
    mode_legend=[Line2D([0],[0],color='#333333',marker=m,ls='',markersize=5,label=l) for _,l,m in MODES]
    fig.legend(handles=mode_legend,loc='outside upper center',ncol=3,frameon=False)
    for ax,field,title in zip(axs,['top1','nll'],['A  Final test top-1 accuracy · higher is better','B  Final test cross-entropy (NLL) · lower is better']):
        for i,key in enumerate(POLICIES):
            for j,(mode,_,marker) in enumerate(MODES):
                row=metrics[key].get(mode)
                if row is None or row[field] is None: continue
                value=row[field]*(100 if field=='top1' else 1)
                pos=i+(j-1)*.22
                ax.scatter(value,pos,color=COLORS[key],marker=marker,s=27,zorder=3)
                ax.annotate(f'{value:.2f}'+('%' if field=='top1' else ''),(value,pos),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
        ax.set_yticks(range(6),[LABELS[k] for k in POLICIES]);ax.invert_yaxis()
        ax.set_title(title);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        ax.set_ylim(5.55,-.55)
        ax.set_xlim((0,100) if field=='top1' else (0,4.3))
        ax.set_xlabel('Accuracy (%)' if field=='top1' else 'Mean −log p(true identity); selected checkpoint only')
        if field=='nll':ax.text(.15,0,'Not recorded for baseline',va='center',fontsize=9,color='#555555')
    save(fig,'final_test_comparison')

    fig,axs=plt.subplots(2,1,figsize=(6.5,5.7),layout='constrained')
    fig.legend(handles=legend_rows,loc='outside upper center',ncol=3,frameon=False)
    for ax,field,title in zip(axs,['train_top1','val_top1'],['A  Training top-1 · policy-specific augmented inputs','B  Validation top-1 · common unaugmented split']):
        for key in POLICIES:
            ax.plot(range(1,31),[100*r[field] for r in curves[key]],color=COLORS[key],ls=STYLES[key],lw=1.8)
            if field=='val_top1':
                epoch=best[key];ax.scatter(epoch,100*curves[key][epoch-1][field],color=COLORS[key],s=22,zorder=4)
        ax.set(title=title,xlabel='Epoch',ylabel='Top-1 accuracy (%)',xlim=(1,30),ylim=(0,100));ax.grid(alpha=.18)
        ax.set_xticks([1,5,10,15,20,25,30])
    save(fig,'all_policy_accuracy')

    fig,axs=plt.subplots(3,2,figsize=(10.5,9.5),layout='constrained')
    for ax,key in zip(axs.flat,POLICIES):
        ax.plot(range(1,31),[r['train_loss'] for r in curves[key]],color=COLORS[key],lw=2,label='Training')
        ax.plot(range(1,31),[r['val_loss'] for r in curves[key]],color=COLORS[key],lw=2,ls='--',label='Validation')
        ax.axvline(best[key],color='#888888',lw=1,ls=':')
        ax.set(title=f'{LABELS[key]} · selected epoch {best[key]}',xlabel='Epoch',ylabel='Margin-adjusted CE',xlim=(1,30.5),ylim=(0,20));ax.grid(alpha=.18)
        ax.legend(loc='upper right',frameon=True,facecolor='white',edgecolor='none',framealpha=1)
    save(fig,'all_policy_diagnostics')

    # One landscape page for meeting discussion: separate axes preserve metric definitions.
    fig,axs=plt.subplots(2,2,figsize=(13.2,8.6),layout='constrained')
    fig.legend(handles=legend,loc='outside upper center',ncol=6,frameon=False,fontsize=10)
    for ax,field,title in zip(axs[0],['train_loss','val_loss'],['A  Training loss · augmented inputs','B  Validation loss · common clean split']):
        for key in POLICIES:
            ax.plot(range(1,31),[r[field] for r in curves[key]],color=COLORS[key],ls=STYLES[key],lw=2)
        ax.set(title=title,xlabel='Epoch',ylabel='Margin-adjusted cross-entropy',xlim=(1,30),ylim=(0,20))
        ax.set_xticks([1,5,10,15,20,25,30]);ax.grid(alpha=.18)
    for ax,field,title in zip(axs[1],['top1','nll'],['C  Final test top-1 · higher is better','D  Final test cross-entropy · lower is better']):
        for i,key in enumerate(POLICIES):
            for j,(mode,_,marker) in enumerate(MODES):
                row=metrics[key].get(mode)
                if row is None or row[field] is None:continue
                value=row[field]*(100 if field=='top1' else 1);pos=i+(j-1)*.22
                ax.scatter(value,pos,color=COLORS[key],marker=marker,s=30,zorder=3)
                ax.annotate(f'{value:.2f}'+('%' if field=='top1' else ''),(value,pos),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
        ax.set_yticks(range(6),[LABELS[k] for k in POLICIES]);ax.set_ylim(5.55,-.55)
        ax.set_title(title);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        ax.set_xlim((0,100) if field=='top1' else (0,4.3))
        ax.set_xlabel('Accuracy (%)' if field=='top1' else 'Mean −ln p(true identity); margin-free')
        if field=='nll':ax.text(.12,0,'Baseline: not recorded',va='center',fontsize=9,color='#555555')
    axs[1,0].legend(handles=mode_legend,loc='lower left',frameon=False,fontsize=9)
    fig.supxlabel('149 identities · 1,498 test crops · one fit per policy. Test panels use selected checkpoints, not epoch series. Baseline TTA was not evaluated.',fontsize=9)
    save(fig,'one_page_overview')

    payload={'report_date':'2026-10-28','revision_date':'2026-10-30','policy_order':POLICIES,'labels':LABELS,'colors':COLORS,
             'curves':curves,'curve_sources':sources,'best_epochs':best,'final_test':metrics,'test_evidence':evidence,
             'test_nll_definition':'Mean natural-log negative true-class probability, computed in float64 with no clipping; probabilities strictly positive. Single view is cached 0° output verified against recorded predictions/confidences.',
             'comparability':'Train/validation CE use the ArcFace label-dependent margin; test NLL uses margin-free probabilities. They are separate quantities and must not share a loss axis. Test NLL has one point per selected checkpoint, not an epoch series.',
             'baseline_missing':'Baseline full probabilities and test loss not saved; no baseline TTA results.',
             'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.png','.pdf')}}
    (OUT/'comparison_data.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (OUT/'final_test_cross_entropy.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['policy','inference','num_test_images','test_top1','test_macro_f1','test_cross_entropy_nll','correct'])
        w.writeheader()
        for key in POLICIES:
            for mode,_,_ in MODES:
                if mode not in metrics[key]:continue
                m=metrics[key][mode];w.writerow({'policy':key,'inference':mode,'num_test_images':1498,'test_top1':m['top1'],'test_macro_f1':m['f1_macro'],'test_cross_entropy_nll':m['nll'] if m['nll'] is not None else 'Not recorded','correct':m['correct']})
    print('All six policies: 30 verified epochs each; 15 cached test probability outputs checked.')
    print('Test NLL by policy:',{k:{m:round(v['nll'],6) if v['nll'] is not None else None for m,v in modes.items()} for k,modes in metrics.items()})
    print('Outputs:',OUT)

if __name__=='__main__':main()
