# COMP 472 Project — Fall 2026
**Group 37 | Concordia University**

## Project Overview
This project compares machine learning algorithms for image recognition and sentiment analysis. The provided datasets contain numerical feature vectors rather than original images or text.

We implemented k-Nearest Neighbours (k-NN), Decision Tree and Random Forest using NumPy, trained a feed-forward neural network with PyTorch, and evaluated k-Means clustering.

## Datasets
**Required before running the code:** Download the Kaggle competition dataset, including the image training/test data and sentiment training/test data, from the [official Kaggle competition page](https://www.kaggle.com/competitions/comp-472-project-fall-2026-concordia/data). Extract all five CSV files into a folder called `data` at the root of this project. The code will not run without these files.

| File | Samples | Description |
|---|---:|---|
| `image_train.csv` | 20,000 | Labeled image features |
| `image_test.csv` | 5,000 | Image features to classify |
| `sentiment_train.csv` | 12,000 | Labeled sentiment embeddings |
| `sentiment_test.csv` | 5,000 | Sentiment embeddings to classify |
| `sample_submission.csv` | 10,000 | Kaggle prediction template |

The original data files and hidden-test predictions are not committed to this repository. To download the official competition data to your local `data/` folder, install the Kaggle CLI (`python -m pip install kaggle`), authenticate with your own Kaggle account and accept the competition rules, then run `python download_data.py`. Alternatively, extract the five CSVs manually into `data/`. Never commit your Kaggle API key.

## Project Files
- `download_data.py` — optional download of the official Kaggle CSVs to a local `data/` folder
- `src/models.py` — from-scratch k-NN, Decision Tree, Random Forest and k-Means
- `src/run_project.py` — model comparisons, validation, neural network training and clustering experiments
- `src/test_project.py` — basic implementation and submission tests
- `create_submission.py` — final Kaggle prediction script containing the weighted cosine image k-NN and five-network sentiment ensemble
- `create_submission_v5.py` — optional alternative image ensemble used for the second Kaggle selection
- `results/` — saved experiment summaries and metrics
- `requirements.txt` — required Python packages

## How to Run
**First, add the Kaggle datasets.** The folder structure must look like this:

```text
COMP472-NN-Group37/
  data/
    image_train.csv
    image_test.csv
    sentiment_train.csv
    sentiment_test.csv
    sample_submission.csv
  src/
  create_submission.py
  create_submission_v5.py
  requirements.txt
```

The `data` folder must be created locally because the datasets are not included in the GitHub repository. Do not place the CSV files inside `src/` or leave them inside the downloaded ZIP. If a CSV is missing, the scripts will raise a `FileNotFoundError`.

After adding the Kaggle files, open a terminal in the project folder:

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
