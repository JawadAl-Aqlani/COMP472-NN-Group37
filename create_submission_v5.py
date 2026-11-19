"""Alternative COMP472 image ensemble submission (Kaggle v5).

This optional candidate uses 3 independently configured, from-scratch weighted
cosine k-NN image classifiers and retains the sentiment predictions from the
final model. It scored 0.639 on the public leaderboard (as reported on Kaggle).

After running python create_submission.py:
    python create_submission_v5.py
Output: results/submission_v5.csv

Only the provided competition feature vectors and training labels are used.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

CONFIGS = [(0.40, 1.0, 11, 12), (0.45, 1.0, 31, 16), (0.50, 1.0, 11, 16)]
BATCH = 250

def predict(Xtr, ytr, Xq, configs=CONFIGS):
    """Average normalized class votes from three weighted cosine k-NN models."""
    results = []
    for power, idf_power, k, vote_power in configs:
        idf = (np.log((len(Xtr) + 1) / ((Xtr > 0).sum(axis=0) + 1)) + 1) ** idf_power
        tr = normalize(np.maximum(Xtr, 0) ** power * idf).astype(np.float32)
        te = normalize(np.maximum(Xq, 0) ** power * idf).astype(np.float32)
        chunks = []
        with threadpool_limits(limits=4):
            for start in range(0, len(te), BATCH):
                sim = te[start:start+BATCH] @ tr.T
                ix = np.argpartition(sim, -k, axis=1)[:, -k:]
                top_sim = np.take_along_axis(sim, ix, axis=1)
                weights = np.maximum(top_sim, 1.e-6) ** vote_power
                votes = np.zeros((len(ix), 50), dtype=np.float32)
                np.add.at(votes, (np.arange(len(ix))[:, None], ytr[ix]), weights)
                votes /= np.maximum(votes.sum(axis=1, keepdims=True), 1e-12)
                chunks.append(votes)
        results.append(np.vstack(chunks))
    return np.mean(results, axis=0).argmax(axis=1).astype(np.int64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--base', default='results/submission.csv',
                        help='Final submission with sentiment labels to reuse')
    parser.add_argument('--output', default='results/submission_v5.csv')
    args = parser.parse_args()

    data = Path(args.data_dir)
    train = pd.read_csv(data / 'image_train.csv')
    test = pd.read_csv(data / 'image_test.csv')
    sample = pd.read_csv(data / 'sample_submission.csv')[['id']]
    previous = pd.read_csv(args.base)
    assert list(previous.columns) == ['id', 'label']
    assert len(previous) == len(sample) == 10000 and previous.id.equals(sample.id)
    assert previous.id.is_unique and previous.label.notna().all()

    columns = [f'feature_{i}' for i in range(500)]
    X = train[columns].to_numpy(dtype=np.float32)
    Xt = test[columns].to_numpy(dtype=np.float32)
    y = train.label.to_numpy(dtype=np.int64)

    tr, va = train_test_split(np.arange(len(y)), test_size=.2,
                              random_state=42, stratify=y)
    val_pred = predict(X[tr], y[tr], X[va])
    accuracy = (val_pred == y[va]).mean()
    print(f'Image ensemble validation accuracy: {accuracy:.5f}', flush=True)

    test_pred = predict(X, y, Xt)
    image = pd.DataFrame({'id': test.id, 'label': test_pred})
    sentiment = previous.loc[previous.id.str.startswith('text_'), ['id', 'label']]
    combined = pd.concat([image, sentiment], ignore_index=True)
    assert len(combined) == 10000 and combined.id.is_unique

    submission = sample.merge(combined, on='id', how='left',
                              validate='one_to_one', sort=False)
    assert submission.id.equals(sample.id) and submission.label.notna().all()
    submission.label = submission.label.astype(np.int64)
    assert submission.loc[submission.id.str.startswith('img_'), 'label'].between(0, 49).all()
    assert submission.loc[submission.id.str.startswith('text_'), 'label'].between(0, 2).all()
    old_sent = previous.id.str.startswith('text_')
    assert submission.loc[old_sent, 'label'].equals(previous.loc[old_sent, 'label'])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    print(f'VERIFIED {out}: {len(submission)} rows, preserved final sentiment predictions')

if __name__ == '__main__':
    main()
