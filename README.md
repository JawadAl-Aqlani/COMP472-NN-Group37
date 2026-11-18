# COMP 472 Project — Fall 2026
**Group 37 | Concordia University**

## Project Overview
This project compares machine learning algorithms on two classification tasks: image recognition and sentiment analysis. The datasets provide numerical feature vectors, so the models work directly with those features rather than raw images or text.

We implemented and tested k-Nearest Neighbours (k-NN), Decision Tree, Random Forest and a feed-forward Neural Network. We also ran a k-Means clustering experiment to study patterns in the data.

## Datasets
The project uses five files from the [COMP 472 Kaggle competition](https://www.kaggle.com/competitions/comp-472-project-fall-2026-concordia/data):

- `image_train.csv` — 20,000 training samples, 500 features, 50 classes
- `image_test.csv` — 5,000 test samples
- `sentiment_train.csv` — 12,000 training samples, 768 features, 3 classes
- `sentiment_test.csv` — 5,000 test samples
- `sample_submission.csv` — submission template

Place these files in the `data/` folder. They are not included in this repository.

## Project Files
- `src/models.py` — implementations of k-NN, Decision Tree, Random Forest and k-Means
- `src/run_project.py` — training, validation, neural network, clustering and baseline predictions
- `src/test_project.py` — basic tests for the implementations and input files
- `create_submission_v3.py` — improved image k-NN model (v3)
- `create_submission_v4.py` — five independently trained sentiment neural networks combined with the v3 image predictions (final Kaggle model)
- `results/` — experiment summaries and evaluation results
- `requirements.txt` — Python dependencies

## How to Run
Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

Check the project files and basic functions:

```powershell
python src/test_project.py
```

Run the experiments and generate the initial predictions:

```powershell
python src/run_project.py
```

Generate the final Kaggle submission:

```powershell
python create_submission_v3.py
python create_submission_v4.py
```

The v3 script creates `results/submission_v3.csv` using improved image k-NN and sentiment predictions from `run_project.py`. The v4 script trains five sentiment neural networks using the supplied training data and combines their averaged outputs with **unchanged v3 image predictions** to create `results/submission_v4.csv`. Run these commands in order. The competition datasets and prediction CSVs are not committed to GitHub.

## Results
The models were compared using an 80/20 stratified training-validation split with random seed 42.

| Task | Selected model | Validation accuracy |
|---|---|---:|
| Image recognition | Weighted cosine k-NN (k = 11) | 59.275% |
| Sentiment analysis (original) | Feed-forward Neural Network | 67.21% |
| Sentiment analysis (v4) | Five-network neural ensemble | 67.625% |

The improved image model applies a square-root transformation and feature weighting before cosine-based neighbour voting. The reported public Kaggle score for our final **v4** submission was **0.639**. This is a public leaderboard result, not a guarantee of the final private leaderboard score. Our earlier v3 submission scored 0.635.

Further results for all required algorithms and the clustering experiment are available in the `results/` folder.

## Team Members
- Jawad Al-Aqlani — 40272579
- Abd Al Rahman Al Kabani — 40247395

## Acknowledgements
We used course materials, Python library documentation and AI assistance for coding, debugging, analysis and presentation preparation. The submitted report describes the project methodology and external assistance in more detail.
