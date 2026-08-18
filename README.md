# PRODIGY_ML_03
Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.

# Cats vs Dogs Classification with SVM (Prodigy InfoTech ML Task 03)

A Support Vector Machine that classifies images as cat or dog, using HOG (Histogram of Oriented Gradients) features extracted from the [Microsoft Cats vs Dogs dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset).

## Overview

The pipeline:
1. Downloads the dataset from Kaggle via the Kaggle API
2. Loads images, converts to grayscale, resizes to 64×64
3. Extracts HOG features (9 orientations, 8×8 pixel cells, 2×2 cell blocks) as the feature representation for each image
4. Splits data into train/test sets (80/20, stratified) and standardizes features
5. Trains an SVM (RBF kernel, `C=10`) on the HOG features
6. Optionally runs a `GridSearchCV` sweep over `C`, `gamma`, and kernel type
7. Evaluates with accuracy, a classification report, and a confusion matrix
8. Visualizes predictions on sample images, color-coded correct (green) / incorrect (red)

## Why HOG + SVM?

SVMs don't operate on raw pixels well for image tasks, so this notebook uses HOG descriptors — a classic computer-vision feature extractor that captures edge/gradient structure — to turn each image into a fixed-length feature vector the SVM can classify.

## Project Structure

```
.
├── PRODIGY_ML_03.ipynb    # Main notebook: data prep, feature extraction, training, evaluation
├── requirements.txt       # Python dependencies
├── README.md
└── LICENSE
```

## Setup

```bash
git clone https://github.com/<your-username>/PRODIGY_ML_03.git
cd PRODIGY_ML_03
pip install -r requirements.txt
```

## Dataset

Dataset: [Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) on Kaggle (~25,000 images, 2 classes).

The notebook was originally written for Google Colab and downloads the dataset directly via the Kaggle API. To reproduce:

1. Get a Kaggle API token: on kaggle.com, go to **Account → Create New API Token**, which downloads `kaggle.json`.
2. **In Colab**: run the "Upload your Kaggle API token" cell and select `kaggle.json` when prompted.
   **Locally**: place `kaggle.json` in `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`, then skip the upload cell.
3. Run the download cell (or manually):
   ```bash
   kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset -p data
   unzip -q data/microsoft-catsvsdogs-dataset.zip -d data
   ```
4. Update `DATA_DIR` in the notebook if your extracted path differs from `data/PetImages`.

**Note:** the dataset contains a handful of corrupted/unreadable files; the loader skips these automatically.

## Usage

Open `PRODIGY_ML_03.ipynb` in Jupyter or Colab and run cells top to bottom.

- `SAMPLES_PER_CLASS` (default `2000`) caps how many images per class are used — set to `None` to train on the full ~12,500 images per class (slower, but more data).
- `run_grid_search` (default `False`) toggles the `GridSearchCV` hyperparameter sweep over `C`, `gamma`, and kernel — set to `True` if you want to search for better parameters (adds significant runtime).
- The final cell (`show_predictions`) displays a handful of sample images with true vs. predicted labels for a quick visual sanity check.

## Results

Exact numbers depend on `SAMPLES_PER_CLASS`, whether grid search was run, and your train/test split — fill in your run's numbers below:

| Metric | Value |
|---|---|
| Test Accuracy | _fill in after running_ |
| Best Params (if grid search used) | _fill in after running_ |

## Requirements

See `requirements.txt`. Core dependencies: scikit-learn, scikit-image, OpenCV, NumPy, Matplotlib, Kaggle API.

## Acknowledgements

- Dataset: [Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset), via Kaggle (originally from the Microsoft Research "Asirra" project)
- Task 03 of the Prodigy InfoTech Machine Learning Internship
