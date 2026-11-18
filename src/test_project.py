"""Fast tests for COMP472 algorithms and Kaggle submission (not a replacement for a full run)."""
from pathlib import Path
import numpy as np
import pandas as pd
from models import KNN, DecisionTree, RandomForest, KMeans

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'results'


def test_algorithms():
    X = np.array([[0.,0.],[0.,0.1],[0.2,0.],[5.,5.],[5.1,5.],[5.,5.2]],dtype=np.float32)
    y = np.array([0,0,0,1,1,1])
    models = [
        ('KNN',KNN(k=1)),
        ('DecisionTree',DecisionTree(max_depth=4,max_features=2,min_samples_split=2,min_samples_leaf=1,n_thresholds=5)),
    ]
    for name, model in models:
        pred=model.fit(X,y).predict(X)
        acc=(pred==y).mean()
        assert acc>=0.8, (name,acc,pred)
        print(f'PASS {name} train-set toy test: {acc:.1%}')
    rng=np.random.default_rng(42)
    Xlarge=np.concatenate([rng.normal(0,.3,(100,2)),rng.normal(5,.3,(100,2))]).astype(np.float32)
    ylarge=np.concatenate([np.zeros(100,dtype=int),np.ones(100,dtype=int)])
    forest=RandomForest(n_estimators=20,max_depth=6,min_samples_leaf=2,max_features=2,n_thresholds=8,max_samples=.8).fit(Xlarge,ylarge)
    forest_acc=(forest.predict(Xlarge)==ylarge).mean()
    assert forest_acc >= .9, forest_acc
    print(f'PASS RandomForest larger toy test: {forest_acc:.1%}')
    clusters = KMeans(k=2).fit_predict(X)
    assert len(clusters)==6 and len(np.unique(clusters))==2
    print('PASS KMeans 2 clusters')


def test_submission():
    sample=pd.read_csv(DATA/'sample_submission.csv')
    sub=pd.read_csv(OUT/'submission.csv')
    assert list(sub.columns)==['id','label']
    assert len(sub)==len(sample)==10000
    assert sub.id.is_unique and not sub.isna().any().any()
    assert sub.id.equals(sample.id), 'Submission ordering / sample IDs differ'
    assert pd.api.types.is_integer_dtype(sub.label)
    is_img=sub.id.str.startswith('img_')
    is_txt=sub.id.str.startswith('text_')
    assert is_img.sum()==is_txt.sum()==5000
    assert (is_img|is_txt).all()
    assert sub.loc[is_img,'label'].between(0,49).all()
    assert sub.loc[is_txt,'label'].between(0,2).all()
    print('PASS Kaggle CSV: 10,000 rows, IDs match, integer labels in range')


def test_data():
    expected={'image_train.csv':20000,'image_test.csv':5000,'sentiment_train.csv':12000,'sentiment_test.csv':5000}
    for fn,n in expected.items():
        f=DATA/fn
        assert f.exists(),f'Missing {f}'
        assert len(pd.read_csv(f,usecols=['id']))==n,(fn,n)
        print(f'PASS {fn[:-4]}: {n:,} rows')
    test_submission()


if __name__=='__main__':
    test_algorithms()
    test_data()
    print('ALL QUICK TESTS PASSED')
