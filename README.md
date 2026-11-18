# COMP 472 — Fall 2026 (Group 37)

Classification of pre-extracted image and sentiment feature vectors using from-scratch k-NN, Decision Tree and Random Forest, a PyTorch neural network, and k-Means clustering.

## Data
Download the five CSV files from the course's official Kaggle competition, and place them inside `data/`: `image_train.csv`, `image_test.csv`, `sentiment_train.csv`, `sentiment_test.csv`, `sample_submission.csv`. Competition data and hidden-test predictions are deliberately not committed to this repository.

## Run
From the repository root:
```powershell
python -m pip install -r requirements.txt
python src/test_project.py
python src/run_project.py
python create_submission_v3.py
```

`run_project.py` evaluates all four supervised algorithm families on both datasets, runs sentiment clustering, and creates baseline predictions plus `results/sentiment_predictions.csv`. Then `create_submission_v3.py` evaluates the power/IDF-weighted cosine k-NN image model and combines its test predictions with the existing sentiment model output into `results/submission_v3.csv`.

The v3 script must run *after* the full baseline script, because it reuses the sentiment prediction file. The baseline script overwrites some outputs. The reported 0.635 public leaderboard score came from the v3 submission and is not independently reproduced by any CSV stored here.

## Validation and methods
The supervised experiments use an 80/20 stratified split with seed 42. Image v3 uses the train-derived inverse-frequency feature weighting, square-root feature transformation, cosine similarity, k=11 and similarity-powered weighted voting. No pretrained classifiers or external data are used.

## Team
Jawad Al-Aqlani (40272579); Abd Al Rahman Al Kabani (40247395); third teammate and contributions to be confirmed before final report.

## External assistance
AI assistance was used for code development, debugging, analysis, and presentation preparation; team members must review, verify and be able to explain the code. The project report should disclose this assistance accurately.
