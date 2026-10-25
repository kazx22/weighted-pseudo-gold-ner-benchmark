
from __future__ import annotations
import csv, json, shutil
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
PUB=ROOT/'results'/'publication'
BRANCHES={
    'bc5cdr': {'label':'BC5CDR','root':ROOT/'results','prefix':''},
    'biored': {'label':'BioRED official','root':ROOT/'results'/'biored','prefix':'biored_'},
    'biored_overlap_excluded': {'label':'BioRED overlap-excluded','root':ROOT/'results'/'biored_overlap_excluded','prefix':'biored_'},
}
MODEL_KEYS=['scispacy','biobert','pubmedbert','clinicalbert','d4data']
MODEL_NAMES={'scispacy':'scispaCy','biobert':'BioBERT','pubmedbert':'PubMedBERT + OpenMed','clinicalbert':'ClinicalBERT','d4data':'D4Data NER (DistilBERT)'}

def load_json(p): return json.loads(p.read_text(encoding='utf-8'))
def write_csv(p, rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows: p.write_text('',encoding='utf-8'); return
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def copy_if(src,dst):
    if src.exists(): dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst); return True
    return False

def eval_rows(root):
    d=load_json(root/'test'/'evaluation_results.json')
    if 'results' in d:
        pseudo=d['results']['pseudo_gold_vs_human']; models=d['results']['models_vs_human']
        norm=lambda x: {'system':x['name'],**{k:x[k] for k in ('precision','recall','f1','tp','fp','fn')}}
    else:
        pseudo=d['pseudo_gold_vs_human']; models=d['models_vs_human']
        norm=lambda x: {'system':x['system'],**{k:x[k] for k in ('precision','recall','f1','tp','fp','fn')}}
    rows=[norm(x) for x in pseudo]+[norm(x) for x in models]
    cawv=root/'test'/'cawv'/'cawv_evaluation.json'
    if cawv.exists():
        m=load_json(cawv)['overall']['CAWV']
        rows.insert(2,{'system':'CAWV',**{k:m[k] for k in ('precision','recall','f1','tp','fp','fn')}})
    return rows

def bar_f1(rows,path,title):
    names=[r['system'] for r in rows]; vals=[float(r['f1']) for r in rows]
    fig,ax=plt.subplots(figsize=(9.5,5.8)); y=range(len(names)); ax.barh(list(y),vals); ax.set_yticks(list(y),names); ax.set_xlim(0,1); ax.set_xlabel('Exact-span micro-F1'); ax.set_title(title); ax.invert_yaxis(); ax.grid(axis='x',alpha=.25); fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=300,bbox_inches='tight'); plt.close(fig)

def kappa(root,tables,figs,label):
    p=root/'test'/'cohen_kappa_results.json'
    if not p.exists(): return False
    d=load_json(p); rows=[]
    for x in d.get('versus_human_gold',[]):
        rows.append({'model':x['model'],'kappa':x['kappa'],'positive_token_precision':x.get('positive_token_precision',''),'positive_token_recall':x.get('positive_token_recall',''),'positive_token_f1':x.get('positive_token_f1','')})
    # BioRED positive-token metrics live in the diagnostic CSV; merge when available.
    diag=root/'test'/'kappa_diagnostic'/'kappa_model_summary.csv'
    if diag.exists():
        with diag.open(encoding='utf-8') as f:
            by={r['model']:r for r in csv.DictReader(f)}
        for r in rows:
            z=by.get(r['model'])
            if z:
                for key in ('positive_token_precision','positive_token_recall','positive_token_f1'):
                    if key in z: r[key]=z[key]
    write_csv(tables/'kappa_vs_human.csv',rows)
    fig,ax=plt.subplots(figsize=(8.5,4.8)); ax.bar([r['model'] for r in rows],[float(r['kappa']) for r in rows]); ax.set_ylim(0,1); ax.set_ylabel("Cohen's kappa"); ax.set_title(f'{label}: token-level agreement with human gold'); ax.tick_params(axis='x',rotation=25); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(figs/'kappa_vs_human.png',dpi=300,bbox_inches='tight'); plt.close(fig); return True

def taxonomy(root,tables,figs,label):
    p=root/'test'/'error_taxonomy'/'taxonomy_summary.json'
    if not p.exists(): return False
    d=load_json(p); rows=[]
    for item in d['models']:
        key=item.get('model_key') or item.get('model')
        totals=item.get('totals') or item.get('counts') or {}
        rows.append({'model':MODEL_NAMES.get(key,item.get('model',key)),'false_positive':totals.get('FALSE_POSITIVE',0),'false_negative':totals.get('FALSE_NEGATIVE',0),'boundary_error':totals.get('BOUNDARY_ERROR',0),'type_confusion':totals.get('TYPE_CONFUSION',0)})
    write_csv(tables/'error_taxonomy.csv',rows)
    fig,ax=plt.subplots(figsize=(10,5.5)); x=list(range(len(rows))); width=.2
    fields=[('false_positive','FP'),('false_negative','FN'),('boundary_error','Boundary'),('type_confusion','Type')]
    for i,(field,nm) in enumerate(fields): ax.bar([v+(i-1.5)*width for v in x],[float(r[field]) for r in rows],width,label=nm)
    ax.set_xticks(x,[r['model'] for r in rows],rotation=25,ha='right'); ax.set_ylabel('Taxonomy event count'); ax.set_title(f'{label}: exact-span error taxonomy'); ax.legend(); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(figs/'error_taxonomy.png',dpi=300,bbox_inches='tight'); plt.close(fig); return True

def method_controls(root,tables,figs,label):
    p=root/'test'/'method_controls'/'method_controls.json'
    if not p.exists(): return False
    d=load_json(p); tu=d['tuned_unweighted']; rows=[
        {'system':'Fixed majority (k=3)','f1':tu['fixed_three_of_five_test_metrics']['f1']},
        {'system':f"DEV-tuned unweighted (k={tu['selected_k']})",'f1':tu['test_metrics']['f1']},
        {'system':'Weighted voting','f1':tu['weighted_test_metrics']['f1']},
    ]
    write_csv(tables/'method_controls_summary.csv',rows)
    write_csv(tables/'lomo_test.csv',d['leave_one_model_out'].get('test_results',[]) if isinstance(d['leave_one_model_out'],dict) else [])
    bar_f1(rows,figs/'method_controls.png',f'{label}: voting controls'); return True

def dev_sensitivity(root,tables,figs,label):
    p=root/'test'/'development_gold_sensitivity'/'development_gold_sensitivity_summary.csv'
    if not p.exists(): return False
    with p.open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    write_csv(tables/'development_gold_sensitivity.csv',rows)
    fig,ax=plt.subplots(figsize=(7.5,4.8)); x=[float(r['percent']) for r in rows]; y=[float(r['mean_test_f1']) for r in rows]; e=[float(r['sd_test_f1']) for r in rows]; ax.errorbar(x,y,yerr=e,marker='o',capsize=4); ax.set_xlabel('Development gold used (%)'); ax.set_ylabel('Mean TEST F1'); ax.set_title(f'{label}: development-label sensitivity'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figs/'development_gold_sensitivity.png',dpi=300,bbox_inches='tight'); plt.close(fig); return True

def relaxed(root,tables,figs,label):
    p=root/'test'/'relaxed_overlap'/'relaxed_overlap_metrics.csv'
    if not p.exists(): return False
    with p.open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    write_csv(tables/'relaxed_overlap.csv',rows)
    fig,ax=plt.subplots(figsize=(8.5,4.8)); ax.bar([r['model'] for r in rows],[float(r['f1']) for r in rows]); ax.set_ylim(0,1); ax.set_ylabel('Relaxed-overlap F1'); ax.set_title(f'{label}: relaxed-overlap evaluation'); ax.tick_params(axis='x',rotation=25); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(figs/'relaxed_overlap.png',dpi=300,bbox_inches='tight'); plt.close(fig); return True

def ablation(root,tables,figs,label):
    p=root/'test'/'cawv'/'ablation'/'test_ablation_metrics.csv'
    if not p.exists(): return False
    with p.open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    write_csv(tables/'cawv_ablation.csv',rows)
    fig,ax=plt.subplots(figsize=(8.5,4.8)); ax.bar([r['method'] for r in rows],[float(r['f1']) for r in rows]); ax.set_ylim(0,1); ax.set_ylabel('Exact-span F1'); ax.set_title(f'{label}: CAWV ablation'); ax.tick_params(axis='x',rotation=25); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(figs/'cawv_ablation.png',dpi=300,bbox_inches='tight'); plt.close(fig); return True

def bootstrap(root,tables,figs,label):
    p=root/'test'/'bootstrap_confidence_intervals.csv'
    if not p.exists(): return False
    with p.open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    cawv=root/'test'/'cawv'/'bootstrap_confidence_intervals.csv'
    if cawv.exists():
        with cawv.open(encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if str(r.get('system','')).upper()=='CAWV': rows.append(r)
    write_csv(tables/'bootstrap_confidence_intervals.csv',rows)
    names=[r['system'] for r in rows]; vals=[float(r['f1']) for r in rows]; lo=[float(r.get('ci_low',r.get('f1_ci_low',r['f1']))) for r in rows]; hi=[float(r.get('ci_high',r.get('f1_ci_high',r['f1']))) for r in rows]
    fig,ax=plt.subplots(figsize=(9.5,6)); y=list(range(len(rows))); ax.errorbar(vals,y,xerr=[[v-l for v,l in zip(vals,lo)],[h-v for v,h in zip(vals,hi)]],fmt='o',capsize=4); ax.set_yticks(y,names); ax.set_xlim(0,1); ax.set_xlabel('Exact-span F1 with 95% bootstrap CI'); ax.set_title(f'{label}: bootstrap confidence intervals'); ax.grid(axis='x',alpha=.25); ax.invert_yaxis(); fig.tight_layout(); fig.savefig(figs/'bootstrap_confidence_intervals.png',dpi=300,bbox_inches='tight'); plt.close(fig); return True

def medgemma(root,tables,figs,label):
    p=root/'test'/'medgemma_evaluation.json'
    if not p.exists(): return False
    d=load_json(p); rows=d.get('results') or []
    rows=[{'system':r['system'],'precision':r['precision'],'recall':r['recall'],'f1':r['f1'],'tp':r['tp'],'fp':r['fp'],'fn':r['fn']} for r in rows]
    # Add CAWV + MedGemma when this branch contains the validated analysis.
    cp=root/'test'/'medgemma_cawv_hybrid'/'cawv_medgemma_comparison.csv'
    if cp.exists():
        with cp.open(encoding='utf-8') as f:
            extra=list(csv.DictReader(f))
        rows.extend([r for r in extra if r['system']=='CAWV + MedGemma'])
    write_csv(tables/'medgemma_comparison.csv',rows)
    bar_f1(rows,figs/'medgemma_comparison.png',f'{label}: MedGemma comparison'); return True

def cawv_subgroups(root,tables,figs,label):
    p=root/'test'/'cawv'/'cawv_evaluation.json'
    if not p.exists(): return False
    d=load_json(p); rows=[]
    for method in ('MV','WV','CAWV'):
        for dim,groups in d['subgroups'][method].items():
            for group,m in groups.items(): rows.append({'method':method,'dimension':dim,'group':group,'f1':m['f1'],'precision':m['precision'],'recall':m['recall']})
    write_csv(tables/'cawv_subgroups.csv',rows)
    groups=[]
    for dim in ('entity_type','span_length','mention_form'):
        for r in rows:
            key=(dim,r['group'])
            if r['dimension']==dim and key not in groups: groups.append(key)
    fig,ax=plt.subplots(figsize=(10,5.5)); x=list(range(len(groups))); width=.25
    for i,method in enumerate(('MV','WV','CAWV')):
        vals=[]
        for dim,g in groups:
            hit=next(r for r in rows if r['method']==method and r['dimension']==dim and r['group']==g); vals.append(float(hit['f1']))
        ax.bar([v+(i-1)*width for v in x],vals,width,label=method)
    ax.set_xticks(x,[g for _,g in groups],rotation=25,ha='right'); ax.set_ylim(0,1); ax.set_ylabel('F1'); ax.set_title(f'{label}: subgroup F1'); ax.legend(); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(figs/'cawv_subgroups.png',dpi=300,bbox_inches='tight'); plt.close(fig); return True

def threshold_figures(root,figs,prefix):
    dev=root/'dev'/'figures'; names={
        f'{prefix}threshold_sensitivity_curve.png':'wv_threshold_sensitivity.png',
        'cawv_alpha_sensitivity.png':'cawv_alpha_sensitivity.png',
        'cawv_threshold_curve_selected_alpha.png':'cawv_threshold_sensitivity.png',
    }
    return {dst:copy_if(dev/src,figs/dst) for src,dst in names.items()}

def generate(allow_partial: bool=False) -> Path:
    PUB.mkdir(parents=True,exist_ok=True); manifest=[]; cross=[]
    for slug,cfg in BRANCHES.items():
        root=cfg['root']; label=cfg['label']; out=PUB/slug; tables=out/'tables'; figs=out/'figures'; tables.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True)
        if not (root/'test'/'evaluation_results.json').exists():
            if allow_partial: continue
            raise FileNotFoundError(root/'test'/'evaluation_results.json')
        rows=eval_rows(root); write_csv(tables/'main_performance.csv',rows); bar_f1(rows,figs/'main_performance.png',f'{label}: held-out exact-span performance');
        for r in rows: cross.append({'setting':label,**r})
        checks={
            'main_performance':True,
            'bootstrap':bootstrap(root,tables,figs,label),
            'kappa':kappa(root,tables,figs,label),
            'taxonomy':taxonomy(root,tables,figs,label),
            'relaxed_overlap':relaxed(root,tables,figs,label),
            'method_controls':method_controls(root,tables,figs,label),
            'development_gold_sensitivity':dev_sensitivity(root,tables,figs,label),
            'cawv_ablation':ablation(root,tables,figs,label),
            'cawv_subgroups':cawv_subgroups(root,tables,figs,label),
            'medgemma':medgemma(root,tables,figs,label),
        }
        checks.update(threshold_figures(root,figs,cfg['prefix']))
        for item,available in checks.items(): manifest.append({'setting':label,'output_family':item,'available':bool(available)})
    crossdir=PUB/'cross_dataset'; write_csv(crossdir/'all_main_performance.csv',cross); write_csv(PUB/'SYMMETRY_MATRIX.csv',manifest)
    # Main-corpus mirror check. Overlap-excluded is a sensitivity branch and may contain deliberate scope exceptions.
    by={(r['setting'],r['output_family']):r['available'] for r in manifest}
    main_families=sorted({r['output_family'] for r in manifest if r['setting']=='BC5CDR'})
    mismatches=[]
    for fam in main_families:
        a=by.get(('BC5CDR',fam),False); b=by.get(('BioRED official',fam),False)
        if a!=b: mismatches.append(fam)
    if mismatches and not allow_partial: raise RuntimeError('Main-corpus reporting asymmetry: '+', '.join(mismatches))
    (PUB/'README.txt').write_text('Canonical symmetric reporting outputs. BC5CDR and BioRED official use the same table/figure families. BioRED overlap-excluded is the separate sensitivity branch.\n',encoding='utf-8')
    print(f'Symmetric publication outputs: {PUB}')
    return PUB

if __name__=='__main__': generate()
