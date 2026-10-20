import json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

def load_qa_dataset():
    try:
        ds = load_dataset('pminervini/HaluEval', 'qa', split='data'); split_used='data'
    except Exception:
        ds = load_dataset('pminervini/HaluEval', 'qa', split='train'); split_used='train'
    required={'knowledge','question','right_answer','hallucinated_answer'}
    missing=required.difference(ds.column_names)
    if missing: raise ValueError(f'Dataset schema mismatch: missing {sorted(missing)}; columns={ds.column_names}')
    return ds, split_used

def make_pairs(ds, n_pairs, seed):
    ds=ds.shuffle(seed=seed).select(range(min(n_pairs,len(ds))))
    rows=[]
    for i,ex in enumerate(ds):
        prompt=('Use the provided knowledge to answer the question.\n'
                f"Knowledge: {ex['knowledge']}\nQuestion: {ex['question']}\nAnswer:")
        for label,key in [(0,'right_answer'),(1,'hallucinated_answer')]:
            rows.append({'pair_id':i,'label':label,'answer_type':'hallucinated' if label else 'reference',
                         'prompt':prompt,'answer':str(ex[key]).strip(),'question':str(ex['question']),
                         'knowledge':str(ex['knowledge'])})
    return pd.DataFrame(rows)

def score_response(model, tokenizer, prompt, answer, device, max_length):
    prompt_ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
    full=tokenizer(prompt+' '+answer,add_special_tokens=False,truncation=True,max_length=max_length,return_tensors='pt')
    ids=full['input_ids'].to(device); mask=full['attention_mask'].to(device)
    prompt_len=len(prompt_ids); seq_len=ids.shape[1]
    if prompt_len>=seq_len:
        # Reserve space for answer tokens if the prompt is unusually long.
        shortened=tokenizer.decode(prompt_ids[-max(1,max_length-16):])
        full=tokenizer(shortened+' '+answer,add_special_tokens=False,truncation=True,max_length=max_length,return_tensors='pt')
        ids=full['input_ids'].to(device); mask=full['attention_mask'].to(device)
        prompt_len=len(tokenizer(shortened+' ',add_special_tokens=False)['input_ids']); seq_len=ids.shape[1]
    if seq_len<2 or prompt_len>=seq_len:
        return {'mean_token_nll':float('nan'),'max_token_nll':float('nan'),'scored_tokens':0}
    with torch.no_grad(): logits=model(input_ids=ids,attention_mask=mask).logits
    # Logits at t predict token t+1. Token positions >= prompt_len are answer tokens.
    log_probs=torch.log_softmax(logits[:,:-1,:].float(),dim=-1)
    targets=ids[:,1:]
    nll=-log_probs.gather(-1,targets.unsqueeze(-1)).squeeze(-1)[0]
    start=max(0,prompt_len-1)
    vals=nll[start:]
    if vals.numel()==0: return {'mean_token_nll':float('nan'),'max_token_nll':float('nan'),'scored_tokens':0}
    return {'mean_token_nll':float(vals.mean().item()),'max_token_nll':float(vals.max().item()),'scored_tokens':int(vals.numel())}

def split_by_pair(df,seed):
    ids=np.array(sorted(df.pair_id.unique())); np.random.default_rng(seed).shuffle(ids)
    n=len(ids); a=int(.6*n); b=int(.8*n)
    tr=set(ids[:a]); va=set(ids[a:b]); te=set(ids[b:])
    df=df.copy(); df['split']=df.pair_id.map(lambda x:'train' if x in tr else ('validation' if x in va else 'test'))
    return df

def at_threshold(y,s,thr):
    pred=(np.asarray(s)>=thr).astype(int)
    return {'threshold':float(thr),'f1':float(f1_score(y,pred,zero_division=0)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'confusion_matrix_tn_fp_fn_tp':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def evaluate(df,col):
    d=df.dropna(subset=[col]); val=d[d.split=='validation']; test=d[d.split=='test']
    yv=val.label.to_numpy(); sv=val[col].to_numpy(); yt=test.label.to_numpy(); st=test[col].to_numpy()
    p,r,ths=precision_recall_curve(yv,sv); f=2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-12)
    thr=float(ths[int(np.nanargmax(f))]) if len(ths) else float(np.median(sv))
    return {'validation_threshold_selection':at_threshold(yv,sv,thr),'test':{
        'n':int(len(yt)),'positive_rate':float(np.mean(yt)),
        'auroc':float(roc_auc_score(yt,st)) if len(np.unique(yt))>1 else None,
        'auprc':float(average_precision_score(yt,st)) if len(np.unique(yt))>1 else None,
        **at_threshold(yt,st,thr)}}

def run_experiment(n_pairs=500,model_name='distilgpt2',output_dir=Path('results/main'),seed=42,max_length=512,batch_size=1):
    started=time.time(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print('Loading HaluEval QA dataset...'); ds,split_used=load_qa_dataset()
    print(f'Loaded {len(ds)} rows, split={split_used}, columns={ds.column_names}')
    df=split_by_pair(make_pairs(ds,n_pairs,seed),seed)
    print(f'{df.pair_id.nunique()} source pairs; {len(df)} candidate answers')
    tokenizer=AutoTokenizer.from_pretrained(model_name); model=AutoModelForCausalLM.from_pretrained(model_name); model.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device); print('Scoring device:',device)
    score_rows=[]
    for i,row in df.iterrows():
        score_rows.append(score_response(model,tokenizer,row.prompt,row.answer,device,max_length))
        if (i+1)%50==0: print(f'Scored {i+1}/{len(df)}')
    df=pd.concat([df.reset_index(drop=True),pd.DataFrame(score_rows)],axis=1)
    df['response_chars']=df.answer.str.len(); df['response_words']=df.answer.str.split().str.len()
    df.to_csv(out/'predictions.csv',index=False)
    results={'dataset':'pminervini/HaluEval','dataset_config':'qa','dataset_split_used':split_used,
             'model_name':model_name,'device':str(device),'n_source_pairs':int(df.pair_id.nunique()),
             'n_candidates':int(len(df)),'seed':seed,'max_length':max_length,
             'split_counts_pairs':{k:int(v) for k,v in df.groupby('split').pair_id.nunique().items()},
             'predictors':{},'note':'Metrics are against HaluEval paired labels; NLL is not a direct probability of truth.'}
    for col in ['mean_token_nll','max_token_nll']: results['predictors'][col]=evaluate(df,col)
    wide=df.pivot(index='pair_id',columns='label',values=['mean_token_nll','max_token_nll','response_words'])
    paired={}
    for col in ['mean_token_nll','max_token_nll','response_words']:
        if (col,0) in wide.columns and (col,1) in wide.columns:
            delta=(wide[(col,1)]-wide[(col,0)]).dropna()
            paired[col+'_hallucinated_minus_reference']={'mean':float(delta.mean()),'median':float(delta.median()),'fraction_positive':float((delta>0).mean()),'n_pairs':int(len(delta))}
    results['paired_descriptive_analysis']=paired; results['runtime_seconds']=round(time.time()-started,2)
    (out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    test=df[df.split=='test'].copy()
    for col,label in [('mean_token_nll','Mean token NLL'),('max_token_nll','Maximum token NLL')]:
        plt.figure(figsize=(7,4))
        for cls,name in [(0,'Reference'),(1,'Hallucinated')]:
            vals=test.loc[test.label==cls,col].dropna(); plt.hist(vals,bins=30,alpha=.55,label=name,density=True)
        plt.xlabel(label); plt.ylabel('Density'); plt.title('Test-set score distributions: '+label); plt.legend(); plt.tight_layout(); plt.savefig(out/(col+'_distribution.png'),dpi=160); plt.close()
    plt.figure(figsize=(7,4))
    for cls,name in [(0,'Reference'),(1,'Hallucinated')]:
        sub=test[test.label==cls]; plt.scatter(sub.response_words,sub.mean_token_nll,alpha=.45,label=name)
    plt.xlabel('Response length (words)'); plt.ylabel('Mean token NLL'); plt.title('Response length and mean token NLL (test set)'); plt.legend(); plt.tight_layout(); plt.savefig(out/'length_vs_mean_nll.png',dpi=160); plt.close()
    thr=results['predictors']['mean_token_nll']['validation_threshold_selection']['threshold']
    test['predicted_hallucination_mean_nll']=(test.mean_token_nll>=thr).astype(int)
    test['error_type_mean_nll']=np.where((test.label==1)&(test.predicted_hallucination_mean_nll==0),'false_negative',np.where((test.label==0)&(test.predicted_hallucination_mean_nll==1),'false_positive','correct'))
    test.to_csv(out/'test_error_analysis.csv',index=False)
    print(json.dumps(results,indent=2)); print('Saved to',out.resolve()); return results
