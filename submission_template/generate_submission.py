import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import pyarrow.parquet as pq
import tqdm
import sys
import train_model

sys.modules["train_model"] = train_model



TEST_PATH = "data/crosstalk_test_inputs.parquet"
fps = ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'MACCS', 'RDK', 'AVALON', 'TOPTOR']


print("Loading model from models/best_model.pkl...")
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
parquet_file = pq.ParquetFile(TEST_PATH)
all_preds = []
all_ids = []

print(f"Generating predictions for all test examples...")
for batch in tqdm.tqdm(parquet_file.iter_batches(batch_size=5000)):
    df_chunk = batch.to_pandas()
    all_ids.extend(df_chunk['RandomID'].values)
    
    test_blocks = []
    for fp in fps:
        parsed = np.array([np.fromstring(x, sep=',', dtype=np.int16) for x in df_chunk[fp]], dtype=np.float32)
        test_blocks.append(sp.csr_matrix(parsed))
    X_chunk = sp.hstack(test_blocks, format='csr')    
    
    pred_probs = model.predict_proba(X_chunk)
    all_preds.extend(pred_probs)
    
print("Saving predictions to submission.csv...")
df_preds = pd.DataFrame({'RandomID': all_ids, 'DELLabel': all_preds})
df_preds.to_csv('submission.csv', index=False)