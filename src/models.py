"""From-scratch COMP472 classifiers. No pretrained classifiers or sklearn estimators."""
import numpy as np

class KNN:
    def __init__(self,k=5,weighted=False,metric='euclidean'):
        self.k,self.weighted,self.metric=k,weighted,metric
    def fit(self,X,y):
        self.X=np.ascontiguousarray(X,dtype=np.float32);self.y=np.asarray(y,dtype=np.int64)
        self.squared=np.einsum('ij,ij->i',self.X,self.X)
        self.classes=int(self.y.max())+1
        return self
    def predict(self,X,batch=128):
        X=np.asarray(X,dtype=np.float32)
        outputs=[]
        for i in range(0,len(X),batch):
            Q=np.ascontiguousarray(X[i:i+batch]);dot=Q@self.X.T
            if self.metric=='cosine':
                qn=np.sqrt(np.einsum('ij,ij->i',Q,Q));tn=np.sqrt(self.squared)
                distances=1-dot/np.maximum(qn[:,None]*tn[None,:],1e-10)
            else:
                distances=np.maximum(np.einsum('ij,ij->i',Q,Q)[:,None]+self.squared[None,:]-2*dot,0)
            nearest=np.argpartition(distances,self.k-1,axis=1)[:,:self.k]
            labels=self.y[nearest]
            weights=1/np.maximum(distances[np.arange(len(Q))[:,None],nearest],1e-8) if self.weighted else np.ones(labels.shape)
            scores=np.zeros((len(Q),self.classes),dtype=np.float64)
            np.add.at(scores,(np.arange(len(Q))[:,None],labels),weights)
            outputs.append(scores.argmax(1))
        return np.concatenate(outputs)

class _Node:
    __slots__=('feature','threshold','left','right','prediction')
    def __init__(self,prediction,feature=-1,threshold=0,left=None,right=None):
        self.prediction=int(prediction);self.feature=feature;self.threshold=threshold;self.left=left;self.right=right

class DecisionTree:
    """CART Gini splitting, NumPy implementation with candidate quantile thresholds."""
    def __init__(self,max_depth=10,min_samples_split=12,min_samples_leaf=5,max_features=None,n_thresholds=12,seed=42):
        self.max_depth=max_depth;self.min_samples_split=min_samples_split;self.min_samples_leaf=min_samples_leaf
        self.max_features=max_features;self.n_thresholds=n_thresholds;self.seed=seed
    def fit(self,X,y):
        self.X=np.asarray(X,dtype=np.float32);self.y=np.asarray(y,dtype=np.int64)
        self.n_classes=int(self.y.max())+1;self.rng=np.random.default_rng(self.seed)
        self.root=self._build(np.arange(len(y)),0);return self
    def _build(self,idx,depth):
        labels=self.y[idx]; counts=np.bincount(labels,minlength=self.n_classes)
        node=_Node(counts.argmax()); n=len(idx)
        if depth>=self.max_depth or n<self.min_samples_split or counts.max()==n or n<2*self.min_samples_leaf:return node
        nfeatures=self.X.shape[1]
        m=nfeatures if self.max_features is None else (max(1,int(np.sqrt(nfeatures))) if self.max_features=='sqrt' else int(self.max_features))
        features=self.rng.choice(nfeatures,min(m,nfeatures),replace=False)
        parent_gini=1-np.dot(counts,counts)/(n*n)
        best_gain=0; best_feature=None;best_threshold=None
        for f in features:
            values=self.X[idx,f]
            thresholds=np.unique(np.quantile(values,np.linspace(.05,.95,self.n_thresholds)))
            for threshold in thresholds:
                left=values<=threshold;nl=int(left.sum());nr=n-nl
                if nl<self.min_samples_leaf or nr<self.min_samples_leaf:continue
                lc=np.bincount(labels[left],minlength=self.n_classes);rc=counts-lc
                gain=parent_gini-(nl/n)*(1-np.dot(lc,lc)/(nl*nl))-(nr/n)*(1-np.dot(rc,rc)/(nr*nr))
                if gain>best_gain:best_gain=gain;best_feature=int(f);best_threshold=float(threshold)
        if best_feature is None:return node
        mask=self.X[idx,best_feature]<=best_threshold
        node.feature=best_feature;node.threshold=best_threshold
        node.left=self._build(idx[mask],depth+1);node.right=self._build(idx[~mask],depth+1)
        return node
    def predict(self,X):
        X=np.asarray(X,dtype=np.float32);out=np.empty(len(X),dtype=np.int64)
        def visit(node,ids):
            if not len(ids):return
            if node.feature<0:out[ids]=node.prediction;return
            mask=X[ids,node.feature]<=node.threshold
            visit(node.left,ids[mask]);visit(node.right,ids[~mask])
        visit(self.root,np.arange(len(X)));return out

class RandomForest:
    """Bootstrap bagging of our own randomized CART trees."""
    def __init__(self,n_estimators=20,max_depth=10,min_samples_leaf=6,max_features='sqrt',n_thresholds=8,max_samples=0.75,seed=42):
        self.n_estimators=n_estimators;self.max_depth=max_depth;self.min_samples_leaf=min_samples_leaf
        self.max_features=max_features;self.n_thresholds=n_thresholds;self.max_samples=max_samples;self.seed=seed
    def fit(self,X,y):
        X=np.asarray(X,dtype=np.float32);y=np.asarray(y,dtype=np.int64);self.n_classes=int(y.max())+1
        rng=np.random.default_rng(self.seed);self.trees=[]
        for n in range(self.n_estimators):
            ids=rng.integers(0,len(y),size=int(len(y)*self.max_samples))
            tree=DecisionTree(max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,n_thresholds=self.n_thresholds,seed=self.seed+n)
            tree.fit(X[ids],y[ids]);self.trees.append(tree)
        return self
    def predict(self,X):
        preds=np.vstack([tree.predict(X) for tree in self.trees]).T
        return np.array([np.bincount(row,minlength=self.n_classes).argmax() for row in preds])

class KMeans:
    """K-means++ initialization and iterative centroid updates, from scratch."""
    def __init__(self,k=3,max_iter=40,seed=42):self.k=k;self.max_iter=max_iter;self.seed=seed
    def fit_predict(self,X):
        X=np.asarray(X,dtype=np.float32);rng=np.random.default_rng(self.seed)
        centroids=[X[rng.integers(len(X))]]
        closest=np.sum((X-centroids[0])**2,axis=1)
        for _ in range(1,self.k):
            idx=rng.choice(len(X),p=closest/closest.sum())
            centroids.append(X[idx]);closest=np.minimum(closest,np.sum((X-X[idx])**2,axis=1))
        C=np.asarray(centroids).copy()
        for _ in range(self.max_iter):
            distances=np.sum(X**2,axis=1)[:,None]+np.sum(C**2,axis=1)[None,:]-2*X@C.T
            labels=distances.argmin(axis=1)
            new=np.vstack([X[labels==j].mean(axis=0) if np.any(labels==j) else C[j] for j in range(self.k)])
            if np.allclose(new,C,rtol=1e-5,atol=1e-6):break
            C=new
        self.centroids=C;self.labels_=labels;return labels
