"""Main training and submission pipeline for the CrossTALK workshop.

Example:
    $ python train_model.py
"""

import os
import time
import random
import warnings
import pickle

import gdown
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
import pyarrow.parquet as pq

import src.dataset
import src.eval
import src.path_and_constants

warnings.filterwarnings("ignore")

_P = src.path_and_constants.Paths()
_C = src.path_and_constants.Constants()

FEATURE_COLS: list[str] = ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'MACCS', 'RDK', 'AVALON', 'TOPTOR']


SEED_MLP = 74

def seed_everything(seed=74):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_data() -> None:
    """Downloads dataset files from Google Drive if they don't exist locally."""
    os.makedirs("data", exist_ok=True)
    for filepath, file_id in _C.file_ids.items():
        if not os.path.exists(filepath):
            print(f"Downloading {filepath}...")
            gdown.download(id=file_id, output=filepath, quiet=False)

# This version of tanimoto simialrity accounts for the fact that some of our fingerprints are not binary
def compute_max_tanimoto(A, B):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A_sum = torch.tensor(np.array(A.power(2).sum(axis=1)).flatten(), dtype=torch.float32, device=device)
    B_sum = torch.tensor(np.array(B.power(2).sum(axis=1)).flatten(), dtype=torch.float32, device=device)
    B_tensor = torch.tensor(B.toarray(), dtype=torch.float32, device=device)
    max_sim = np.zeros(A.shape[0], dtype=np.float32)
    chunk_size = 2000
    for i in tqdm.tqdm(range(0, A.shape[0], chunk_size), desc="Computing Max Tanimoto to Test Set"):
        A_chunk = torch.tensor(A[i:i+chunk_size].toarray(), dtype=torch.float32, device=device)
        A_sum_chunk = A_sum[i:i+chunk_size]
        intersection = torch.matmul(A_chunk, B_tensor.T)
        denom = A_sum_chunk.unsqueeze(1) + B_sum.unsqueeze(0) - intersection
        sim = intersection / torch.clamp(denom, min=1e-6)
        max_sim[i:i+A_chunk.shape[0]] = sim.max(dim=1)[0].cpu().numpy()
    return max_sim


def get_or_create_features():
    print("Loading data...")
    df_train = pd.read_parquet(_P.train_path)
    df_test = pd.read_parquet(_P.test_path)
    y_train = df_train['DELLabel'].values.astype(np.float32)
    
    train_blocks = []
    print("Processing training fingerprints...")
    for fp in FEATURE_COLS:
        parsed = np.array([np.fromstring(x, sep=',', dtype=np.int16) for x in df_train[fp]], dtype=np.float32)
        train_blocks.append(sp.csr_matrix(parsed))
    X_train_full = sp.hstack(train_blocks, format='csr')
    
    test_blocks = []
    print("Processing test fingerprints...")
    for fp in FEATURE_COLS:
        parsed = np.array([np.fromstring(x, sep=',', dtype=np.int16) for x in df_test[fp]], dtype=np.float32)
        test_blocks.append(sp.csr_matrix(parsed))
    X_test_full = sp.hstack(test_blocks, format='csr')
    
    fp_train_ecfp4 = train_blocks[0]
    fp_test_ecfp4 = test_blocks[0]
    
    print("Fitting SelectKBest...")
    pos_indices = (y_train == 1)
    pos_bit_counts = np.asarray(X_train_full[pos_indices].sum(axis=0)).ravel()
    keep_mask_active = (pos_bit_counts >= 1)
    
    X_train_filtered_1 = X_train_full[:, keep_mask_active]
    X_test_filtered_1 = X_test_full[:, keep_mask_active]
    
    n_kept_active = np.sum(keep_mask_active)
    k_val = min(5000, n_kept_active)
    selector = SelectKBest(score_func=chi2, k=k_val)
    X_train_filtered_2 = selector.fit_transform(X_train_filtered_1, y_train)
    X_test_filtered_2 = selector.transform(X_test_filtered_1)
    
    X_combined_filtered = sp.vstack([X_train_filtered_2, X_test_filtered_2], format='csr')
    
    print("Fitting TruncatedSVD...")
    svd = TruncatedSVD(n_components=512, random_state=42)
    reduced_features = svd.fit_transform(X_combined_filtered)
    
    max_test_sim = compute_max_tanimoto(fp_train_ecfp4, fp_test_ecfp4)
    
    return reduced_features, max_test_sim, keep_mask_active, selector, svd, y_train


class TabularDataset(Dataset):
    def __init__(self, X, y, weights):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
    def __len__(self): 
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]


class PredictorMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )
    def forward(self, x): 
        return self.net(x).squeeze(-1)

#this allows for the model to be used like an sklearn model, without seperate loading of data preprocessing objects
class CrosstalkPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, keep_mask_active, selector, svd, model_state_dict):
        self.keep_mask_active = keep_mask_active
        self.selector = selector
        self.svd = svd
        self.model_state_dict = model_state_dict
        self._model = None
        
    def _init_model(self):
        if self._model is None:
            self._model = PredictorMLP(input_dim=512)
            self._model.load_state_dict(self.model_state_dict)
            self._model.eval()
            
    def predict_proba(self, X):
        """
        X should be a CSR matrix of the combined fingerprints.
        Returns the probability of class 1.
        """
        self._init_model()
        
        X_filtered = X[:, self.keep_mask_active]
        X_selected = self.selector.transform(X_filtered)
        X_reduced = self.svd.transform(X_selected)
        
        X_tensor = torch.tensor(X_reduced, dtype=torch.float32)
        with torch.no_grad():
            pred_logit = self._model(X_tensor)
            pred_prob = torch.sigmoid(pred_logit).numpy()
            
        return pred_prob

    def predict(self, X):
        """Returns binary predictions with a 0.5 threshold."""
        return (self.predict_proba(X) >= 0.5).astype(int)
        
    def fit(self, X, y):
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_model' in state:
            del state['_model']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None

CrosstalkPipeline.__module__ = "train_model"


def main() -> None:
    download_data()
    seed_everything(SEED_MLP)
    
    df_train_meta = pd.read_parquet(_P.train_path, columns=['DELLabel'])
    df_test_meta = pd.read_parquet(_P.test_path, columns=['RandomID'])
    test_ids = df_test_meta['RandomID'].values
    
    reduced_features, max_test_sim, keep_mask_active, selector, svd, y_train = get_or_create_features()
    
    X_train_reduced = reduced_features[:len(df_train_meta)]
    X_test_reduced = reduced_features[len(df_train_meta):]
    
    sample_weights = np.ones(len(y_train), dtype=np.float32)
    pos_mask = (y_train == 1)
    neg_mask = (y_train == 0)
    num_pos = pos_mask.sum()
    num_neg = neg_mask.sum()
    class_weight_pos = num_neg / num_pos
    
    
    sample_weights[pos_mask] = class_weight_pos * (1.0 + 10.0 * (max_test_sim[pos_mask] ** 2)) # upweight positives a lot,
    sample_weights[neg_mask] = 1.0 * (1.0 + 3.0 * (max_test_sim[neg_mask] ** 2)) # upweight negatives a little
    sample_weights = sample_weights / sample_weights.mean()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PredictorMLP(input_dim=X_train_reduced.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train_reduced, y_train, sample_weights, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # Need to do this to make sure dataloader is consistent
    g = torch.Generator()
    g.manual_seed(SEED_MLP)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    train_dataset = TabularDataset(X_tr, y_tr, w_tr)
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, worker_init_fn=seed_worker, generator=g)
    
    val_dataset = TabularDataset(X_val, y_val, w_val)
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_steps = 0
        for X_b, y_b, w_b in train_dataloader:
            X_b, y_b, w_b = X_b.to(device), y_b.to(device), w_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss = (loss * w_b).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1
            
        model.eval()
        val_loss = 0
        val_steps = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_b, y_b, w_b in val_dataloader:
                X_b, y_b, w_b = X_b.to(device), y_b.to(device), w_b.to(device)
                logits = model(X_b)
                loss = criterion(logits, y_b)
                loss = (loss * w_b).mean()
                val_loss += loss.item()
                val_steps += 1
                
                preds = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(y_b.cpu().numpy())
                
        val_preds_all = np.concatenate(val_preds)
        val_targets_all = np.concatenate(val_targets)
    
    metrics = src.eval.evaluate(val_targets_all, val_preds_all)        
    print("\nLocal Validation Split Metrics (with 95% Confidence Intervals):")
    for name, res in metrics.items():
        print(
            f"  {name:25s}: {res['val']:.4f} "
            f"(95% CI: [{res['lower']:.4f}, {res['upper']:.4f}])"
        )
            
    # # Save raw PyTorch model
    # os.makedirs("models", exist_ok=True)
    # torch.save(model.state_dict(), 'models/mlp_model.pt')
    # print("Saved MLP model state to models/mlp_model.pt")
            
    model.eval()
    test_preds = []
    chunk_size = 5000
    with torch.no_grad():
        for i in range(0, X_test_reduced.shape[0], chunk_size):
            X_test_chunk = torch.tensor(X_test_reduced[i:i+chunk_size], dtype=torch.float32).to(device)
            preds = torch.sigmoid(model(X_test_chunk)).cpu().numpy()
            test_preds.append(preds)
            
    test_preds = np.concatenate(test_preds)
    
    # Extract state dict on CPU so the model is portable
    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    
    print(f"\nGenerating submission file {_P.submission_path}")
    df_preds = pd.DataFrame({'RandomID': test_ids, 'DELLabel': test_preds})
    df_preds.to_csv(_P.submission_path, index=False)
    
    #Package into the CrosstalkPipeline and save
    pipeline = CrosstalkPipeline(
        keep_mask_active=keep_mask_active,
        selector=selector,
        svd=svd,
        model_state_dict=cpu_state_dict
    )
    
    os.makedirs(os.path.dirname(_P.model_path), exist_ok=True)
    print(f"Saving combined model to {_P.model_path}...")
    with open(_P.model_path, 'wb') as f:
        pickle.dump(pipeline, f)    

if __name__ == "__main__":
    import sys
    sys.modules["train_model"] = sys.modules[__name__]
    main()
