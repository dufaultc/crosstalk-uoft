# Submission Template

This is a template to submit your final model and code. We will need to replicate your model exactly so we can ultimately use it to make predictions on new chemicals.

Please write your model summary below and make sure your script runs.

---

## Model Writeup (1-2 paragraphs)

Write a brief summary of your approach. Make sure you cover:
- What molecular fingerprint representation you chose (e.g. AVALON, ECFP6, MACCS) and why.
- Your model choice, hyperparameter tuning, and local validation metrics (e.g. ROC-AUC, Hits@k).

---

## Structure

```text
submission_template/
├── README.md             <- This file (contains your writeup)
├── requirements.txt      <- Package requirements
├── train_model.py        <- Training script (saves your model and generates predictions)
├── data/
│   └── README.md         <- Dataset setup instructions
├── models/
│   └── best_model.pkl    <- Your final trained model (save it here)
└── src/
    ├── __init__.py
    ├── dataset.py        <- Data loaders
    └── eval.py           <- Evaluation functions
```

---

## How to Run

1. **Install packages**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Place datasets** in `data/` (see `data/README.md` for download links).
3. **Run the script**:
   ```bash
   python train_model.py
   ```
   This script will run your local validation with confidence intervals, train the final model on all data, save it to `models/best_model.pkl`, and output `submission.csv`.
