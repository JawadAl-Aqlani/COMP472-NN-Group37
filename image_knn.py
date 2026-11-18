"""Image classification with weighted cosine k-nearest neighbours.

Uses only the provided image feature vectors. Fits IDF weights on training data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

POWER = 0.5
K = 11
VOTE_POWER = 12
BATCH = 500


def compute_idf(x):
    n = x.shape[0]
    return (np.log((n + 1) / ((x > 0).sum(axis=0) + 1)) + 1).astype(np.float32)


def transform(x, idf):
    return normalize(np.power(np.maximum(x, 0), POWER) * idf).astype(np.float32)


def knn_predict(train_x, train_y, test_x, k=K):
    result = []
    with threadpool_limits(limits=4):
        for start in range(0, len(test_x), BATCH):
            scores = test_x[start:start + BATCH] @ train_x.T
            neighbors = np.argpartition(scores, -k, axis=1)[:, -k:]
            top_scores = np.take_along_axis(scores, neighbors, axis=1)
            weights = np.maximum(top_scores, 1.e-6) ** VOTE_POWER
            votes = np.zeros((len(neighbors), 50), dtype=np.float32)
            labels = train_y[neighbors]
            row_ids = np.arange(len(neighbors))
            for j in range(k):
                np.add.at(votes, (row_ids, labels[:, j]), weights[:, j])
            result.append(votes.argmax(axis=1))
    return np.concatenate(result)


def predict_image_test(data_dir='data', validate=True):
    """Return DataFrame with id,label predictions for image_test.csv."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / 'image_train.csv')
    test = pd.read_csv(data_dir / 'image_test.csv')
    features = [f'feature_{j}' for j in range(500)]
    x = train[features].to_numpy(dtype=np.float32)
    xt = test[features].to_numpy(dtype=np.float32)
    y = train.label.to_numpy(dtype=np.int64)

    if validate:
        a, b = train_test_split(
            np.arange(len(y)), test_size=.2, random_state=42, stratify=y
        )
        idf = compute_idf(x[a])
        pred = knn_predict(transform(x[a], idf), y[a], transform(x[b], idf))
        accuracy = (pred == y[b]).mean()
        print(f'Image held-out validation accuracy: {accuracy:.5f}', flush=True)

    full_idf = compute_idf(x)
    test_pred = knn_predict(
        transform(x, full_idf), y, transform(xt, full_idf)
    )
    return pd.DataFrame({'id': test.id, 'label': test_pred})


if __name__ == '__main__':
    predictions = predict_image_test()
    out = Path('results/image_predictions.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    print(f'Saved {out} ({len(predictions)} rows)')
