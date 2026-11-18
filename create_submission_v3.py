"""COMP472: from-scratch weighted cosine k-NN image inference.
Run from anywhere after putting this file inside COMP472_Project/.
Existing sentiment predictions must be created by the original run_project.py.
No pretrained models, no external training data, no hidden test labels.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data';OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
POWER=0.5
K=11
VOTE_POWER=12
BATCH=500

def compute_idf(x):
    n=x.shape[0]
    return (np.log((n+1)/((x>0).sum(axis=0)+1))+1).astype(np.float32)

def transform(x,idf):
    return normalize(np.power(np.maximum(x,0),POWER)*idf).astype(np.float32)

def knn_predict(train_x,train_y,test_x,k=K):
    result=[]
    with threadpool_limits(limits=4):
        for start in range(0,len(test_x),BATCH):
            scores=test_x[start:start+BATCH]@train_x.T
            neighbors=np.argpartition(scores,-k,axis=1)[:,-k:]
            top_scores=np.take_along_axis(scores,neighbors,axis=1)
            weights=np.maximum(top_scores,1.e-6)**VOTE_POWER
            votes=np.zeros((len(neighbors),50),dtype=np.float32)
            labels=train_y[neighbors]
            row_ids=np.arange(len(neighbors))
            for j in range(k):
                np.add.at(votes,(row_ids,labels[:,j]),weights[:,j])
            result.append(votes.argmax(axis=1))
    return np.concatenate(result)

def load(task,split):
    df=pd.read_csv(DATA/f'{task}_{split}.csv')
    x=df[[f'feature_{j}' for j in range(500)]].to_numpy(dtype=np.float32)
    return df,x

def main():
    train,x=load('image','train');test,xt=load('image','test')
    y=train.label.to_numpy(dtype=np.int64)
    a,b=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
    idf=compute_idf(x[a]); pred=knn_predict(transform(x[a],idf),y[a],transform(x[b],idf))
    acc=(pred==y[b]).mean();print(f'Image held-out validation accuracy: {acc:.5f}')
    full_idf=compute_idf(x)
    test_pred=knn_predict(transform(x,full_idf),y,transform(xt,full_idf))
    image=pd.DataFrame({'id':test.id,'label':test_pred})
    image.to_csv(OUT/'image_predictions_v3.csv',index=False)
    sentiment_path=OUT/'sentiment_predictions.csv'
    if not sentiment_path.exists():
        raise FileNotFoundError(f'Missing {sentiment_path}. Run original src/run_project.py first.')
    sentiment=pd.read_csv(sentiment_path)
    sample=pd.read_csv(DATA/'sample_submission.csv')[['id']]
    combined=pd.concat([image,sentiment],ignore_index=True)
    assert len(combined)==len(sample)==10000 and combined.id.is_unique
    sub=sample.merge(combined,on='id',how='left',validate='one_to_one',sort=False)
    assert sub.id.equals(sample.id) and sub.label.notna().all()
    sub['label']=sub['label'].astype(int)
    assert sub.loc[sub.id.str.startswith('img_'),'label'].between(0,49).all()
    assert sub.loc[sub.id.str.startswith('text_'),'label'].between(0,2).all()
    sub.to_csv(OUT/'submission_v3.csv',index=False)
    print(f'VERIFIED submission_v3.csv ({len(sub)} rows; IDs match template; labels valid)')
if __name__=='__main__':main()
