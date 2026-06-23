# CrossTALK Model Submission Template

Welcome to the CrossTALK workshop model submission repository template! This repository is designed to help your team submit your best trained models, the code to reproduce your results, and a brief write-up of your approach.

---

## 📝 Model Writeup (Action Required)

*Replace this section with a 1-2 paragraph summary of your team's approach. Make sure to cover the following topics:*

### Preprocessing & Feature Engineering
- *Which molecular representation/fingerprint did you choose (e.g. AVALON, ECFP6, MACCS) and why?*
- *Did you perform any dimensionality reduction (like UMAP/PCA) or feature selection?*

### Model Choice & Tuning
- *What model type did you use (e.g. XGBoost, CatBoost, RandomForest) and how did you tune its hyperparameters?*

### Validation Strategy & Results
- *How did you structure your validation (e.g. 5-fold cross-validation)?*
- *What were your local cross-validation scores (ROC-AUC, Precision@K, Hits@K, etc.)?*

---

## 📁 Repository Structure

We follow a pruned version of the **Cookiecutter Data Science** project layout:

```text
crosstalk_template/
├── README.md             <- This file (contains your team's writeup)
├── requirements.txt      <- Packages required to run the pipeline
├── train_model.py        <- The main script to train, evaluate, and save your model
├── data/
│   └── README.md         <- Instructions for dataset placement
├── models/
│   └── best_model.pkl    <- Your final serialized model (deposit here)
└── src/
    ├── __init__.py
    ├── dataset.py        <- Data utilities (parquet dataloader)
    └── eval.py           <- Custom evaluation metrics (BinaryEvaluator)
```

---

## 🚀 Getting Started

### 1. Environment Setup
To replicate the environment and install dependencies, run:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
Follow the instructions in [data/README.md](data/README.md) to place your `crosstalk_train.parquet` and `crosstalk_test_inputs.parquet` files.

### 3. Run Training and Submission Generation
Execute the main script to download the datasets (if missing), perform cross-validation, save the trained model, and generate your submission file:
```bash
python train_model.py
```
This will output:
- Your trained model saved to `models/best_model.pkl`.
- Your predictions saved to `submission.csv` in the root directory (ready to upload to the Kaggle platform).
