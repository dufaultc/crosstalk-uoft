import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import pyarrow.parquet as pq

TEST_PATH = "data/crosstalk_test_inputs.parquet"
fps = ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'MACCS', 'RDK', 'AVALON', 'TOPTOR']

parquet_file = pq.ParquetFile(TEST_PATH)
first_batch = next(parquet_file.iter_batches(batch_size=10))
example = first_batch.to_pandas()

with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
test_blocks = []
for fp in fps:
    parsed = np.array([np.fromstring(x, sep=',', dtype=np.int16) for x in example[fp]], dtype=np.float32)
    test_blocks.append(sp.csr_matrix(parsed))
X_test_full = sp.hstack(test_blocks, format='csr')    


pred_probs = model.predict_proba(X_test_full)

print(f"Predicted probability of DELLabel=1: {pred_probs[0:10]}")