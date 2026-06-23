"""Main training and submission pipeline for the CrossTALK workshop.

Example:
    $ python train_model.py
"""

import os
import pickle
import gdown
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn.model_selection
import xgboost as xgb

import src.dataset
import src.eval
import src.path_and_constants

_P = src.path_and_constants.Paths()
_C = src.path_and_constants.Constants()

# Configure molecular features to load. You can specify a single column or list of columns.
# Options: 'ATOMPAIR', 'MACCS', 'ECFP6', 'ECFP4', 'FCFP4', 'FCFP6', 'TOPTOR', 'RDK', 'AVALON'
FEATURE_COLS: list[str] | str = ["AVALON"]


def download_data() -> None:
    """Downloads dataset files from Google Drive if they don't exist locally."""
    os.makedirs("data", exist_ok=True)
    for filepath, file_id in _C.file_ids.items():
        if not os.path.exists(filepath):
            print(f"Downloading {filepath}...")
            gdown.download(id=file_id, output=filepath, quiet=False)


def main() -> None:
    """Executes local validation, full training, and submission generation."""
    download_data()

    print(f"\nLoading training data with features '{FEATURE_COLS}'")
    X, y = src.dataset.load_data(_P.train_path, FEATURE_COLS)

    # Local validation split
    X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Local split evaluation
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

    print("Evaluating model performance on local validation split...")
    y_pred_proba = val_model.predict_proba(X_val)[:, 1]
    metrics = src.eval.evaluate(y_val, y_pred_proba)

    print("\nLocal Validation Split Metrics (with 95% Confidence Intervals):")
    for name, res in metrics.items():
        print(
            f"  {name:25s}: {res['val']:.4f} "
            f"(95% CI: [{res['lower']:.4f}, {res['upper']:.4f}])"
        )

    # Train final model on 100% of data
    print("\nTraining final model on all data...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(_P.model_path), exist_ok=True)
    print(f"Saving final model to {_P.model_path}")
    with open(_P.model_path, "wb") as f:
        pickle.dump(final_model, f)

    # Generate submission file
    print(f"\nLoading test data with features '{FEATURE_COLS}'")
    X_test = src.dataset.load_data(_P.test_path, FEATURE_COLS, y_col=None)
    test_probs = final_model.predict_proba(X_test)[:, 1]

    print(f"Generating submission file {_P.submission_path}")
    pf = pq.ParquetFile(_P.test_path)
    df = pf.read(columns=["RandomID"]).to_pandas()
    df["DELLabel"] = test_probs
    df.to_csv(_P.submission_path, index=False)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
