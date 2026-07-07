# Team DELtheASMS! submission

## Model Writeup (1-2 paragraphs)

We trained a 3-layer MLP predictor (512-> 512, 512->256, 256->1) with GELU activation, batch normalization, and dropout, using 8 of 9 fingerprints (ATOMPAIR was not used) for our input. Data preprocessing is performed as follows for training: all fingerprints are concatenated into a large feature array, filtering out any features which are zero across all training set positives. Next, Chi-square feature selection is used to select 5000 features which are likely to be relevant to predicting the activity label. Finally, the feature reduced train and test dataset are concatenated and TruncatedSVD dimensionality reduction is used to reduce the dimensionality of the entire combined dataset to 512 features.\
Our predictor MLP is then trained on 90% of the training data (10% randomly chosen for validation set, stratified by the activity label), with training set examples being sample weighted based upon their maximum Tanimoto similarity (calculated using the ECPF4 fingerprint only) to test set examples. Training set positives are upweighted much more than training set negatives.

Rationale:
* All fingerprints (except ATOMPAIR) were used to provide the model with the most information possible about the molecule. We did not see improved performance by narrowing down the fingerprints based on our knowledge of what they capture. ATOMPAIR was not included primarily to increase training speed and because we did not see performance improvement when it was included.
* We chose to performed combined dimensionality reduction of the train+test dataset in order to create features which distinguish examples in both datasets rather than just the train dataset. This makes our approach transductive. Truncated SVD was chosen for this because it handles our large and sparse input better than other aproaches.
* An MLP was chosen for the predictor as it was suitable for handling the dense, continuous features we had for our examples after performing dimensionality reduction.  Hyperparameters were mostly chosen based on what seemed reasonable for this model and dataset, and adjusted based on what led to improved validation AUPRC. Training epochs was adjusted to be 50, as this led to maximum validation set AUPRC performance of ~0.885
* This model achieved 8 hits @ 200, with AUPRC performance of ~0.885 and AUC of ~0.98 on our 10% Validation set.


---

## Structure

```text
submission_template/
├── README.md             <- This file (contains our writeup)
├── requirements.txt      <- Package requirements
├── train_model.py        <- Training script (saves our model and generates predictions)
├── usage.py              <- Example usage script
├── data/
│   └── README.md         <- Dataset setup instructions
├── models/
│   └── best_model.pkl    <- our final trained model (save it here)
└── src/
    ├── __init__.py
    ├── dataset.py        <- Data loaders (not used)
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
   This script will run your local validation with confidence intervals, train the final model on all data, save it to `models/best_model.pkl`, and output `submission.csv`. We package the pytorch model as well as the objects we create for data preprocessing into a single wrapper, which contains methods so that the model can be used like an sklearn model.


`usage.py` demonstrates how the saved model can be loaded and new predictions can be made.
