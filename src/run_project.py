"""Run required COMP472 comparisons, errors/clustering, and make verified Kaggle CSV."""
import os,sys,json,time,random,platform,copy
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,adjusted_rand_score,normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler,normalize
from models import KNN,DecisionTree,RandomForest,KMeans
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data';OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
SEED=42
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(min(4,os.cpu_count() or 2))
FAST=os.environ.get('FAST','0')=='1'

def load(task,kind):
    df=pd.read_csv(DATA/f'{task}_{kind}.csv')
    feats=[f'feature_{i}' for i in range(500 if task=='image' else 768)]
    return df,df[feats].to_numpy(dtype=np.float32)

class MLP(nn.Module):
    def __init__(self,n_input,n_classes,dropout=.25):
        super().__init__();self.net=nn.Sequential(nn.Linear(n_input,256),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,n_classes))
    def forward(self,x):return self.net(x)

def neural_fit(X,y,Xv=None,yv=None,epochs=15,lr=.001,seed=SEED):
    torch.manual_seed(seed);model=MLP(X.shape[1],int(y.max())+1);optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
    X_t=torch.from_numpy(np.ascontiguousarray(X));y_t=torch.from_numpy(np.asarray(y,dtype=np.int64))
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_t,y_t),batch_size=256,shuffle=True)
    best=-1;best_weights=None;history=[]
    for ep in range(epochs):
        model.train();total=0
        for bx,by in loader:
            optimizer.zero_grad();loss=nn.functional.cross_entropy(model(bx),by);loss.backward();optimizer.step();total+=loss.item()*len(by)
        va=None
        if Xv is not None:
            va=accuracy_score(yv,neural_predict(model,Xv))
            if va>best:best=va;best_weights=copy.deepcopy(model.state_dict())
        history.append({'epoch':ep+1,'train_loss':round(total/len(y),5),'validation_accuracy':va})
    if best_weights is not None:model.load_state_dict(best_weights)
    return model,history,int(np.argmax([h['validation_accuracy'] for h in history])+1) if Xv is not None else epochs

def neural_predict(model,X):
    model.eval();outs=[]
    with torch.no_grad():
        for i in range(0,len(X),512):outs.append(model(torch.from_numpy(np.ascontiguousarray(X[i:i+512]))).argmax(1).numpy())
    return np.concatenate(outs)

def scaler_fit(X_train,X_others):
    scaler=StandardScaler().fit(X_train)
    return scaler.transform(X_train).astype(np.float32),[scaler.transform(z).astype(np.float32) for z in X_others]

def task_experiment(task):
    df,X=load(task,'train');test_df,Xtest=load(task,'test');y=df.label.to_numpy(dtype=np.int64)
    train_idx,val_idx=train_test_split(np.arange(len(y)),test_size=.2,random_state=SEED,stratify=y)
    xt,xv=X[train_idx],X[val_idx];yt,yv=y[train_idx],y[val_idx]
    xt_std,[xv_std]=scaler_fit(xt,[xv]);results=[];predictions={};config={}
    def assess(name,params,pred,elapsed):
        score=accuracy_score(yv,pred)
        results.append({'dataset':task,'model':name,'params':json.dumps(params),'validation_accuracy':round(score,6),'seconds':round(elapsed,2)})
        predictions[(name,json.dumps(params))]=pred
        print(task,name,params,'accuracy',round(score,4),'seconds',round(elapsed,1),flush=True)
        return score
    # Classical algorithms, implemented locally from scratch
    for metric in ('euclidean','cosine'):
        for k in (1,3,7,15):
            start=time.time();model=KNN(k=k,weighted=True,metric=metric).fit(xt_std,yt)
            pred=model.predict(xv_std);assess('KNN',{'k':k,'weighted':True,'metric':metric,'scale':'standard'},pred,time.time()-start)
    for metric in ('cosine','euclidean'):
        for k in (1,3,5):
            start=time.time();model=KNN(k=k,weighted=True,metric=metric).fit(xt,yt)
            pred=model.predict(xv);assess('KNN',{'k':k,'weighted':True,'metric':metric,'scale':'raw'},pred,time.time()-start)
    for depth in (6,12):
        start=time.time();model=DecisionTree(max_depth=depth,min_samples_leaf=8,max_features=64 if task=='image' else 80,n_thresholds=8,seed=SEED).fit(xt,yt)
        pred=model.predict(xv);assess('DecisionTree',{'max_depth':depth,'min_samples_leaf':8,'max_features':64 if task=='image' else 80,'n_thresholds':8},pred,time.time()-start)
    for trees in ((8,20) if not FAST else (8,)):
        start=time.time();model=RandomForest(n_estimators=trees,max_depth=12,min_samples_leaf=6,n_thresholds=6,max_samples=.7).fit(xt,yt)
        pred=model.predict(xv);assess('RandomForest',{'n_estimators':trees,'max_depth':12,'min_samples_leaf':6,'n_thresholds':6,'max_samples':.7},pred,time.time()-start)
    for lr in (.001,.0003):
        start=time.time();model,hist,best_epoch=neural_fit(xt_std,yt,xv_std,yv,epochs=10 if FAST else 18,lr=lr)
        pred=neural_predict(model,xv_std);assess('NeuralNetwork',{'learning_rate':lr,'best_epoch':best_epoch,'dropout':.25},pred,time.time()-start)
        pd.DataFrame(hist).to_csv(OUT/f'{task}_neural_lr_{lr}_history.csv',index=False)
    res=pd.DataFrame(results);res.to_csv(OUT/f'{task}_experiments.csv',index=False)
    best=res.sort_values('validation_accuracy',ascending=False).iloc[0]
    best_params=json.loads(best['params']);best_name=best['model'];print('BEST',task,best_name,best_params,flush=True)
    best_pred=predictions[(best_name,best['params'])]
    cm=confusion_matrix(yv,best_pred,labels=np.arange(int(y.max())+1));pd.DataFrame(cm).to_csv(OUT/f'{task}_confusion_matrix.csv',index=False)
    report=classification_report(yv,best_pred,output_dict=True,zero_division=0)
    with open(OUT/f'{task}_classification_report.json','w') as f:json.dump(report,f,indent=2)
    # Clustering: sentiment is designated clustering task, no labels used for fit
    if task=='sentiment':
        # use 4000 held-out feature vectors for fast reproducible 3-vs-6 comparison
        sample=np.random.default_rng(SEED).choice(len(X),size=4000,replace=False)
        scaled=StandardScaler().fit_transform(X[sample]).astype(np.float32)
        cluster_results=[]
        for k in (3,6):
            clusters=KMeans(k=k,seed=SEED).fit_predict(scaled)
            cluster_results.append({'k':k,'adjusted_rand_index':adjusted_rand_score(y[sample],clusters),
                                    'normalized_mutual_information':normalized_mutual_info_score(y[sample],clusters)})
            pd.crosstab(pd.Series(y[sample],name='true_label'),pd.Series(clusters,name='cluster')).to_csv(OUT/f'clustering_k{k}_crosstab.csv')
        pd.DataFrame(cluster_results).to_csv(OUT/'clustering_results.csv',index=False)
    # Fit the selected hyperparameters on ALL labeled training data, never hidden test labels
    if best_name=='KNN':
        fullX,[tX]=scaler_fit(X,[Xtest]) if best_params['scale']=='standard' else (X,[Xtest]); final_model=KNN(k=best_params['k'],weighted=True,metric=best_params['metric']).fit(fullX,y);test_pred=final_model.predict(tX)
    elif best_name=='DecisionTree':
        final_model=DecisionTree(**{k:v for k,v in best_params.items() if k in ('max_depth','min_samples_leaf','max_features','n_thresholds')}).fit(X,y);test_pred=final_model.predict(Xtest)
    elif best_name=='RandomForest':
        final_model=RandomForest(**best_params).fit(X,y);test_pred=final_model.predict(Xtest)
    else:
        fullX,[tX]=scaler_fit(X,[Xtest]);final_model,_,_=neural_fit(fullX,y,epochs=best_params['best_epoch'],lr=best_params['learning_rate'])
        test_pred=neural_predict(final_model,tX);torch.save(final_model.state_dict(),OUT/f'{task}_final_nn.pt')
    pd.DataFrame({'id':test_df.id,'label':test_pred}).to_csv(OUT/f'{task}_predictions.csv',index=False)
    return {'dataset':task,'selected_model':best_name,'parameters':best_params,'validation_accuracy':float(best['validation_accuracy'])}

def make_submission():
    sample=pd.read_csv(DATA/'sample_submission.csv')[['id']]
    p=pd.concat([pd.read_csv(OUT/'image_predictions.csv'),pd.read_csv(OUT/'sentiment_predictions.csv')],ignore_index=True)
    assert len(sample)==len(p)==10000 and p.id.is_unique
    merged=sample.merge(p,on='id',how='left',validate='one_to_one',sort=False)
    assert merged.label.notna().all()
    merged.label=merged.label.astype(int)
    assert merged.id.equals(sample.id)
    assert merged.loc[merged.id.str.startswith('img_'),'label'].between(0,49).all()
    assert merged.loc[merged.id.str.startswith('text_'),'label'].between(0,2).all()
    merged.to_csv(OUT/'submission.csv',index=False)
    print('VERIFIED submission.csv',len(merged),'rows',flush=True)

if __name__=='__main__':
    results=[task_experiment(t) for t in ('image','sentiment')]
    make_submission()
    with open(OUT/'summary.json','w') as f:
        json.dump({'selections':results,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'seed':SEED,'validation_split':.2},f,indent=2)
