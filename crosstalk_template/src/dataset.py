"""Module for handling dataset loading for the crosstalk template project."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse
from tqdm.auto import tqdm

FINGERPRINT_TYPES = [
    "ATOMPAIR",
    "MACCS",
    "ECFP6",
    "ECFP4",
    "FCFP4",
    "FCFP6",
    "TOPTOR",
    "RDK",
    "AVALON",
]

def basic_dataloader(
    filepath: str,
    x_col: str,
    y_col: str = "DELLabel",
    max_to_load: int = None,
    chunk_size: int = 20000,
):
    """Loads data from a Parquet file into memory as a sparse matrix.

    Args:
        filepath: Path to the Parquet file.
        x_col: Name of the feature column (one of the fingerprint types).
        y_col: Name of the label column. Set to None to only load features (e.g. for test set).
        max_to_load: Maximum number of rows to load. If None, loads all rows.
        chunk_size: Number of rows to read at a time from disk to control memory usage.

    Returns:
        X: Scipy sparse matrix of features.
        y: Numpy array of labels if y_col is provided, otherwise only X is returned.
    """
    if x_col not in FINGERPRINT_TYPES:
        raise ValueError(
            f"Invalid fingerprint type: {x_col}. Supported types: {FINGERPRINT_TYPES}"
        )

    pf = pq.ParquetFile(filepath)
    columns = [x_col] + ([y_col] if y_col is not None else [])
    
    total_rows = pf.metadata.num_rows
    if max_to_load is None or max_to_load > total_rows:
        max_to_load = total_rows

    mats = []
    y_list = []
    loaded = 0

    n_chunks = int(np.ceil(max_to_load / chunk_size))
    pbar = tqdm(total=n_chunks, desc="Loading chunks")

    for batch in pf.iter_batches(columns=columns, batch_size=min(chunk_size, max_to_load)):
        batch_df = pa.Table.from_batches([batch]).to_pandas()
        remaining = max_to_load - loaded
        if len(batch_df) > remaining:
            batch_df = batch_df.iloc[:remaining]
            
        # Convert comma-separated string features to matrix
        exploded = batch_df[x_col].str.split(",", expand=True).astype(float, copy=False)
        mats.append(scipy.sparse.csr_matrix(exploded))
        
        if y_col is not None:
            y_list.append(batch_df[y_col].values)
            
        loaded += len(batch_df)
        del batch_df, exploded
        pbar.update(1)
        if loaded >= max_to_load:
            break

    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()

    X = scipy.sparse.vstack(mats)
    if y_col is not None and y_list:
        y = np.concatenate(y_list)
        return X, y
    return X
