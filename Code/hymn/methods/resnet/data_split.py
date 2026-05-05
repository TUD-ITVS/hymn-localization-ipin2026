import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from hymn.config import RESNET, PROCESSED_DATA_DIR

TEST_SPLIT = RESNET["test_split"]

def _downsample(df, key, common_ids, target_counts):
    # Filter out IDs that aren't common
    df_common = df[df[key].isin(common_ids)].copy()
    
    # Mapping each row to the target n's for its point_id
    df_common['target_n'] = df_common[key].map(target_counts)
    
    # Shuffling the dataframe first ensures random selection
    df_common['rank'] = df_common.sample(frac=1, random_state=42).groupby(key).cumcount()
    
    # Filter to keep only the required rows, clean up temp columns, and sort
    df_equal = (
        df_common[df_common['rank'] < df_common['target_n']]
        .drop(columns=['target_n', 'rank'])
        .sort_values('ts')
        .reset_index(drop=True)
    )
    return df_equal

def match_dataframe_sizes(ble, uwb, wifi, key: str = "point_id"):
    # Downsamples both ble and uwb so that for each point_id,
    # both dataframes have the same number of rows (the minimum
    # of the two).

    # Get counts for each point_id in both
    ble_counts = ble[key].value_counts()
    uwb_counts = uwb[key].value_counts()
    wifi_counts = wifi[key].value_counts()

    # Find common point_ids
    common_ids = set(ble_counts.index).intersection(uwb_counts.index).intersection(wifi_counts.index)

    # Find the target count for each point_id (minimum of the two)
    target_counts = {
        mid: min(ble_counts[mid], uwb_counts[mid], wifi_counts[mid]) for mid in common_ids}
    
    ble_equal = _downsample(ble, key, common_ids, target_counts)
    uwb_equal = _downsample(uwb, key, common_ids, target_counts)
    wifi_equal = _downsample(wifi, key, common_ids, target_counts)

    return ble_equal, uwb_equal, wifi_equal


def build_Xy(ble: Optional, uwb: Optional, wifi: Optional): # type: ignore
    if ble is not None and uwb is not None and wifi is not None:
        # match and align sizes by point_id
        ble_m, uwb_m, wifi_m = match_dataframe_sizes(ble, uwb, wifi, 'point_id')
        X = np.concatenate((
            np.array(ble_m['residual_range_map'].tolist()),
            np.array(uwb_m['residual_range_map'].tolist()),
            np.array(wifi_m['residual_range_map'].tolist())
        ), axis=1)
        y = np.array(ble_m[['ref_x', 'ref_y']].values.tolist()).astype(
            np.float32)
        return X, y
    elif ble is not None:
        X = np.array(ble['residual_range_map'].tolist())
        y = np.array(ble[['ref_x', 'ref_y']].values.tolist()
                     ).astype(np.float32)
        return X, y
    elif uwb is not None:
        X = np.array(uwb['residual_range_map'].tolist())
        y = np.array(uwb[['ref_x', 'ref_y']].values.tolist()
                     ).astype(np.float32)
        return X, y
    elif wifi is not None:
        X = np.array(wifi['residual_range_map'].tolist())
        y = np.array(wifi[['ref_x', 'ref_y']].values.tolist()
                     ).astype(np.float32)
        return X, y
    else:
        raise ValueError('At least one of ble or uwb must be provided')


def split_random(ble=None, uwb=None, wifi=None, test_size=TEST_SPLIT, seed=42, transpose=True, aug=False):
    # Random epoch-level split: every reference point appears in both train and test (interpolation regime).
    out_dir = os.path.join(PROCESSED_DATA_DIR, "random_split")
    os.makedirs(out_dir, exist_ok=True)
    aug_label = '_aug' if aug else ''
    
    # Map inputs and determine if the dataset is single or fused
    inputs = {'ble': ble, 'uwb': uwb, 'wifi': wifi}
    active_techs = {k: v for k, v in inputs.items() if v is not None}
    is_fused = len(active_techs) == 3
    
    if is_fused:
        ble, uwb, wifi = match_dataframe_sizes(ble, uwb, wifi, 'point_id')
        active_techs = {'ble': ble, 'uwb': uwb, 'wifi': wifi}
        tech_label = 'fused'
    else:
        tech_label = list(active_techs.keys())[0]

    # Extract unique point IDs and sample splits grouped by ID for all active sensor dataframes
    ids = list(active_techs.values())[0]['point_id'].unique()
    
    train_dfs = {k: pd.concat([df[df['point_id'] == mid].sample(frac=1-test_size, random_state=seed) for mid in ids]) for k, df in active_techs.items()}
    test_dfs = {k: pd.concat([df[df['point_id'] == mid].sample(frac=test_size, random_state=seed) for mid in ids]) for k, df in active_techs.items()}

    # Drop image data to process tabular metadata
    train_meta_list = [df.drop('residual_range_map', axis=1).reset_index(drop=True) for df in train_dfs.values()]
    test_meta_list = [df.drop('residual_range_map', axis=1).reset_index(drop=True) for df in test_dfs.values()]
    
    train_meta = pd.concat(train_meta_list, axis=1).reset_index(drop=True) if is_fused else train_meta_list[0]
    test_meta = pd.concat(test_meta_list, axis=1).reset_index(drop=True) if is_fused else test_meta_list[0]

    # Save processed metadata formats to disk
    for meta, split in zip([train_meta, test_meta], ['train', 'test']):
        base_path = os.path.join(out_dir, f"{tech_label}_{split}{aug_label}")
        meta.to_csv(f"{base_path}.csv", index=False)
        meta.to_pickle(f"{base_path}.pkl")

    # Build feature arrays and targets using dictionary extraction (defaults to None if missing)
    X_train, y_train = build_Xy(train_dfs.get('ble'), train_dfs.get('uwb'), train_dfs.get('wifi'))
    X_test, y_test = build_Xy(test_dfs.get('ble'), test_dfs.get('uwb'), test_dfs.get('wifi'))

    if transpose:
        return X_train.transpose(0, 2, 3, 1), X_test.transpose(0, 2, 3, 1), y_train, y_test
    
    return X_train, X_test, y_train, y_test
    

def split_leave_points_out(ble=None, uwb=None, wifi=None, test_size=TEST_SPLIT, seed=42, transpose=True, aug=False):
    # Leave-points-out spatial hold-out: entire reference points held out of training (extrapolation regime).
    out_dir = os.path.join(PROCESSED_DATA_DIR, "spatial_holdout")
    os.makedirs(out_dir, exist_ok=True)
    aug_label = '_aug' if aug else ''
    
    # Determine mode (fused vs single) and standardize datasets into a list
    is_fused = all(df is not None for df in [ble, uwb, wifi])
    
    if is_fused:
        ble, uwb, wifi = match_dataframe_sizes(ble, uwb, wifi, 'point_id')
        dfs = [ble, uwb, wifi]
        tech = 'fused'
    else:
        dfs = [df for df in [ble, uwb, wifi] if df is not None]
        tech = 'ble' if ble is not None else 'uwb' if uwb is not None else 'wifi'

    # Split point IDs into train and test groups
    ids = dfs[0]['point_id'].unique()
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=42)
    
    train_dfs = [df[df['point_id'].isin(train_ids)] for df in dfs]
    test_dfs = [df[df['point_id'].isin(test_ids)] for df in dfs]

    # Process tabular metadata (exclude 'residual_range_map')
    if is_fused:
        train_meta = pd.concat([df.drop('residual_range_map', axis=1) for df in train_dfs], axis=1).reset_index(drop=True).dropna()
        test_meta = pd.concat([df.drop('residual_range_map', axis=1) for df in test_dfs], axis=1).reset_index(drop=True).dropna()
    else:
        train_meta = train_dfs[0].drop('residual_range_map', axis=1)
        test_meta = test_dfs[0].drop('residual_range_map', axis=1)
        
    # Save metadata to disk via iteration
    for meta, split_name in zip([train_meta, test_meta], ['train', 'test']):
        base_path = os.path.join(out_dir, f"{tech}_{split_name}{aug_label}")
        meta.to_csv(f"{base_path}.csv", index=False)
        meta.to_pickle(f"{base_path}.pkl")

    # Extract Targets (y) and Features (X)
    # y is safely extracted from the primary dataset (BLE in fused, or the single provided tech)
    y_train = np.array(train_dfs[0][['ref_x', 'ref_y']].values.tolist(), dtype=np.float32)
    y_test = np.array(test_dfs[0][['ref_x', 'ref_y']].values.tolist(), dtype=np.float32)

    # Use list comprehension to parse images into numpy arrays
    X_train_list = [np.array(df['residual_range_map'].tolist()) for df in train_dfs]
    X_test_list = [np.array(df['residual_range_map'].tolist()) for df in test_dfs]

    X_train = np.concatenate(X_train_list, axis=1) if is_fused else X_train_list[0]
    X_test = np.concatenate(X_test_list, axis=1) if is_fused else X_test_list[0]

    # Shuffle and reshape
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=seed)

    if transpose:
        X_train, X_test = X_train.transpose(0, 2, 3, 1), X_test.transpose(0, 2, 3, 1)
        
    return X_train, X_test, y_train, y_test
