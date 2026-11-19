# COMP 472 Project — Fall 2026
**Group 37 | Concordia University**

## Project Overview
This project compares machine learning algorithms for image recognition and sentiment analysis. The provided datasets contain numerical feature vectors rather than original images or text.

We implemented k-Nearest Neighbours (k-NN), Decision Tree and Random Forest using NumPy, trained a feed-forward neural network with PyTorch, and evaluated k-Means clustering.

## Datasets
Download the five competition CSV files from [Kaggle](https://www.kaggle.com/competitions/comp-472-project-fall-2026-concordia/data) and place them in the `data/` folder.

| File | Samples | Description |
|---|---:|---|
| `image_train.csv` | 20,000 | Labeled image features |
| `image_test.csv` | 5,000 | Image features to classify |
| `sentiment_train.csv` | 12,000 | Labeled sentiment embeddings |
| `sentiment_test.csv` | 5,000 | Sentiment embeddings to classify |
| `sample_submission.csv` | 10,000 | Kaggle prediction template |

The original data files and hidden-test predictions are not committed to this repository.

## Project Files
- `src/models.py` — from-scratch k-NN, Decision Tree, Random Forest and k-Means
- `src/run_project.py` — model comparisons, validation, neural network training and clustering experiments
- `src/test_project.py` — basic implementation and submission tests
- `create_submission.py` — final Kaggle prediction script containing the weighted cosine image k-NN and five-network sentiment ensemble
- `create_submission_v5.py` — optional alternative image ensemble used for the second Kaggle selection
- `results/` — saved experiment summaries and metrics
- `requirements.txt` — required Python packages

## How to Run
From the project folder:

```powershell
python -m pip install -r requirements.txt
python src/run_project.py
python src/test_project.py
python create_submission.py
```

The final command creates **`results/submission.csv`** with 10,000 predictions matched to Kaggle's sample IDs. It uses only the supplied training data to fit the image classifier and the sentiment neural networks. Running the project requires the five CSV files in `data/`.

To generate the alternative v5 submission after the final script, run `python create_submission_v5.py`. This creates `results/submission_v5.csv` while reusing the sentiment predictions in `results/submission.csv`.

## Results
We used an 80/20 stratified train-validation split with random seed 42.

| Task | Final approach | Validation accuracy |
|---|---|---:|
| Image recognition | Weighted cosine k-NN (k=11) | 59.275% |
| Sentiment analysis | Five-network PyTorch ensemble | 67.625% |

The image classifier applies square-root feature transformation and training-derived feature weights before cosine neighbour voting. The sentiment classifier averages the outputs of five separately trained neural networks.

**Reported public Kaggle score: 0.639.** This is the score of our selected competition submission, not a result from the hidden private leaderboard. Re-training can produce slightly different results depending on the environment.

Full comparison experiments for the required algorithms and clustering are in the `results/` folder.

## Team Members
- Jawad Al-Aqlani — 40272579
- Abd Al Rahman Al Kabani — 40247395

## Acknowledgements
Course materials, Python documentation, and AI tools were used to support code development and debugging.
