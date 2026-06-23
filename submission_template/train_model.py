"""Main training and submission script for the CrossTALK workshop."""

import os
import pickle
import gdown
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.dataset import basic_dataloader
from src.eval import evaluate_predictions

# ----------------------------------------------------------------------
# 1. Config & File Downloads
# ----------------------------------------------------------------------
TRAIN_PATH = "data/crosstalk_train.parquet"
TEST_PATH = "data/crosstalk_test_inputs.parquet"
MODEL_PATH = "models/best_model.pkl"
SUBMISSION_PATH = "submission.csv"

# Choose feature representation
# Options: 'ATOMPAIR', 'MACCS', 'ECFP6', 'ECFP4', 'FCFP4', 'FCFP6', 'TOPTOR', 'RDK', 'AVALON'
FEATURE_COL = "AVALON"

# Google Drive File IDs from the workshop materials
FILE_IDS = {
    TRAIN_PATH: "11S5p0QgP1X9rOFiIjNSLydLenJwm7hle",
    TEST_PATH: "15iMvnmIraM-geCI-vG9iR5naliWfh5tA",
}


def download_data_if_missing():
    """Downloads dataset files from Google Drive if they don't exist locally."""
    os.makedirs("data", exist_ok=True)
    for filepath, file_id in FILE_IDS.items():
        if not os.path.exists(filepath):
            print(f"Downloading {filepath} from Google Drive...")
            gdown.download(id=file_id, output=filepath, quiet=False)
        else:
            print(f"Found local data file: {filepath}")


def main():
    # Make sure datasets are available
    download_data_if_missing()

    # ----------------------------------------------------------------------
    # 2. Load the Training Data
    # ----------------------------------------------------------------------
    print(f"\n--- Loading training data with feature '{FEATURE_COL}' ---")
    # For speed/testing you can set max_to_load to a small number, e.g., 5000
    X_train_full, y_train_full = basic_dataloader(
        filepath=TRAIN_PATH,
        x_col=FEATURE_COL,
        y_col="DELLabel",
        max_to_load=None,  # Load full dataset
    )
    print(f"Train features shape: {X_train_full.shape}")
    print(f"Train labels shape: {y_train_full.shape}")

    # ----------------------------------------------------------------------
    # 3. Model Definition & Local Validation Evaluation with CIs
    # ----------------------------------------------------------------------
    print("\n--- Splitting data for local validation (80% train, 20% validation) ---")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )
    print(f"Local Train shape: {X_tr.shape}, Local Val shape: {X_val.shape}")

    print("\n--- Training model on local train split ---")
    # Define hyper-parameters (customize these!)
    params = {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }
    
    val_model = xgb.XGBClassifier(**params)
    val_model.fit(X_tr, y_tr)

    print("Predicting on local validation split...")
    y_pred_proba_val = val_model.predict_proba(X_val)[:, 1]

    # Compute metrics & confidence intervals
    print("Computing metrics and 95% bootstrap confidence intervals...")
    metrics_results = evaluate_predictions(
        y_true=y_val,
        y_pred_proba=y_pred_proba_val,
        threshold=0.5,
        compute_ci=True,
        ci=0.95,
        n_iterations=1000,
    )

    print("\nLocal Validation Split Performance (with 95% Confidence Intervals):")
    for name, res in metrics_results.items():
        val = res["val"]
        lower = res["lower"]
        upper = res["upper"]
        print(f"  {name:25s}: {val:.4f}  (95% CI: [{lower:.4f}, {upper:.4f}])")

    # ----------------------------------------------------------------------
    # 4. Train on ALL training data and Save (Pickle)
    # ----------------------------------------------------------------------
    print("\n--- Training Final Model on 100% of Training Data ---")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X_train_full, y_train_full)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Saving final model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)

    # ----------------------------------------------------------------------
    # 5. Load Test Data & Generate Predictions
    # ----------------------------------------------------------------------
    print("\n--- Loading Test Inputs ---")
    X_test = basic_dataloader(
        filepath=TEST_PATH,
        x_col=FEATURE_COL,
        y_col=None,
        max_to_load=None,
    )
    print(f"Test features shape: {X_test.shape}")

    print("Predicting probabilities on test set using the final model...")
    test_probs = final_model.predict_proba(X_test)[:, 1]

    # Read the RandomID column from the test file for submission alignment
    print("Preparing submission CSV...")
    pf = pq.ParquetFile(TEST_PATH)
    submission_df = pf.read(columns=["RandomID"]).to_pandas()
    submission_df["DELLabel"] = test_probs

    # Output to CSV
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print("\nDone! Ready for submission.")


if __name__ == "__main__":
    main()
