"""COMP472 v4: five-network sentiment ensemble plus unchanged v3 image predictions.

Uses only the provided competition CSVs. All models are trained from scratch.
Run from repository root after producing results/submission_v3.csv:
    python create_submission_v4.py
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED=42
CONFIGS=[
    {'seed':42,'hidden':[256,128],'dropout':0.25,'weight_decay':0.001,'epochs':6},
    {'seed':55,'hidden':[256,128,64],'dropout':0.20,'weight_decay':0.003,'epochs':10},
    {'seed':68,'hidden':[256,128],'dropout':0.25,'weight_decay':0.001,'epochs':8},
    {'seed':81,'hidden':[256,128,64],'dropout':0.20,'weight_decay':0.003,'epochs':8},
    {'seed':94,'hidden':[256,128],'dropout':0.25,'weight_decay':0.001,'epochs':7},
]

def model_for(config):
    layers=[];n=768
    for hidden in config['hidden']:
        layers += [nn.Linear(n,hidden),nn.ReLU(),nn.Dropout(config['dropout'])]
        n=hidden
    layers += [nn.Linear(n,3)]
    return nn.Sequential(*layers)

def train_model(X,y,config):
    torch.manual_seed(config['seed'])
    model=model_for(config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=config['weight_decay'])
    train=torch.utils.data.TensorDataset(torch.from_numpy(np.array(X,dtype=np.float32,copy=True)),torch.from_numpy(np.array(y,dtype=np.int64,copy=True)))
    loader=torch.utils.data.DataLoader(train,batch_size=256,shuffle=True)
    for epoch in range(config['epochs']):
        model.train()
        for xx,yy in loader:
            optimizer.zero_grad()
            loss=nn.functional.cross_entropy(model(xx),yy)
            loss.backward();optimizer.step()
    return model

def logits_for(model,X):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(X),512):
            xx=torch.from_numpy(np.array(X[start:start+512],dtype=np.float32,copy=True))
            out.append(model(xx).numpy())
    return np.concatenate(out)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',default='data')
    parser.add_argument('--v3',default='results/submission_v3.csv')
    parser.add_argument('--out',default='results/submission_v4.csv')
    args=parser.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    data=Path(args.data_dir)
    train_df=pd.read_csv(data/'sentiment_train.csv')
    test_df=pd.read_csv(data/'sentiment_test.csv')
    sample=pd.read_csv(data/'sample_submission.csv')[['id']]
    original=pd.read_csv(args.v3)
    assert list(original.columns)==['id','label']
    assert len(original)==len(sample)==10000 and original.id.equals(sample.id)
    assert original.id.is_unique and original.label.notna().all()
    cols=[f'feature_{i}' for i in range(768)]
    X=train_df[cols].to_numpy(dtype=np.float32)
    Xt=test_df[cols].to_numpy(dtype=np.float32)
    y=train_df.label.to_numpy(dtype=np.int64)
    # Validation: models share an 80/20 split; fit scaling only to training.
    tr,va=train_test_split(np.arange(len(y)),test_size=.2,random_state=SEED,stratify=y)
    scaler=StandardScaler().fit(X[tr])
    xtr=scaler.transform(X[tr]).astype(np.float32);xval=scaler.transform(X[va]).astype(np.float32)
    val_logits=[]
    for index,c in enumerate(CONFIGS):
        model=train_model(xtr,y[tr],c)
        lg=logits_for(model,xval);val_logits.append(lg)
        print('validation model',index+1,'accuracy',round(accuracy_score(y[va],lg.argmax(1)),5),flush=True)
    pred_val=np.mean(val_logits,axis=0).argmax(axis=1)
    val_acc=accuracy_score(y[va],pred_val)
    print('Ensemble sentiment validation accuracy',round(val_acc,5),flush=True)
    # Retrain on all labelled training rows; never train on test labels.
    scaler_full=StandardScaler().fit(X)
    xfull=scaler_full.transform(X).astype(np.float32);xtest=scaler_full.transform(Xt).astype(np.float32)
    test_logits=[]
    for index,c in enumerate(CONFIGS):
        model=train_model(xfull,y,c)
        test_logits.append(logits_for(model,xtest))
        print('Full training model',index+1,'complete',flush=True)
    sentiment_prediction=np.mean(test_logits,axis=0).argmax(axis=1).astype(int)
    sentiment=pd.DataFrame({'id':test_df.id,'label':sentiment_prediction})
    original_images=original.loc[original.id.str.startswith('img_'),['id','label']]
    combined=pd.concat([original_images,sentiment],ignore_index=True)
    assert combined.id.is_unique and len(combined)==10000
    final=sample.merge(combined,on='id',validate='one_to_one',how='left',sort=False)
    assert final.id.equals(sample.id) and final.label.notna().all()
    final['label']=final['label'].astype(int)
    assert final.loc[final.id.str.startswith('img_'),'label'].between(0,49).all()
    assert final.loc[final.id.str.startswith('text_'),'label'].between(0,2).all()
    images=final.id.str.startswith('img_')
    assert final.loc[images,'label'].equals(original.loc[images,'label'])
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);final.to_csv(out,index=False)
    meta={'sentiment_validation_accuracy':val_acc,'ensemble_count':len(CONFIGS),'configurations':CONFIGS,'seed':SEED,'base_image_submission':str(args.v3),'output':str(out)}
    out.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2))
    print('VERIFIED:',out,len(final),'rows; unchanged v3 image predictions',flush=True)

if __name__=='__main__':main()
