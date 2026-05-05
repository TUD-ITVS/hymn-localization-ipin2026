"""HYMN dataset ingestion. Reads raw CSVs, applies anchor-name normalization
and position remap so downstream pipelines (ResNet + classical) all see
canonical anchor labels matching the surveyed positions table.

Anchor remap rationale: raw WIFI_01..06 follow the Aruba SSID numbering and
raw BLE_01..05 follow the Metirionic beacon IDs; both disagree with the
surveyed-position labels in anchor_coordinates.csv. Root cause and
two-method verification: ICRA/wifi_mapping_report.md.
"""
import ast
import os

import numpy as np
import pandas as pd

from hymn.config import DATA_DIR
from hymn.preprocessing import compute_true_ranges, compute_residual_ranges


WIFI_ANCHOR_REMAP = {
    "WIFI_01": "WIFI_03",
    "WIFI_02": "WIFI_02",
    "WIFI_03": "WIFI_04",
    "WIFI_04": "WIFI_05",
    "WIFI_05": "WIFI_01",
    "WIFI_06": "WIFI_06",
}
BLE_ANCHOR_REMAP = {
    "BLE_01": "BLE_03",
    "BLE_02": "BLE_04",
    "BLE_03": "BLE_02",
    "BLE_04": "BLE_05",
    "BLE_05": "BLE_01",
}


def _normalize_ble_anchor(aid):
    """Raw BLE measurement CSVs use 'BLE1'..'BLE5'; anchor table uses 'BLE_01'..'BLE_05'."""
    if aid.startswith("BLE") and not aid.startswith("BLE_"):
        return "BLE_0" + aid[3:]
    return aid


def _remap_ble_ids(ids):
    return [BLE_ANCHOR_REMAP.get(_normalize_ble_anchor(a), _normalize_ble_anchor(a)) for a in ids]


def _remap_wifi_ids(ids):
    return [WIFI_ANCHOR_REMAP.get(a, a) for a in ids]


def read_data(sensor, file_type='csv'):
    fmt = 'pkl' if file_type == 'pickle' else 'parquet' if file_type == 'parquet' else 'csv'
    data_file = os.path.join(DATA_DIR, file_type, f'{sensor}.{fmt}')
    if file_type == 'pickle':
        return pd.read_pickle(data_file)
    elif file_type == 'parquet':
        return pd.read_parquet(data_file)
    elif file_type == 'csv':
        df = pd.read_csv(data_file)
        # CSV stores list-valued columns as their Python repr; parse back to list.
        # `nan` is not valid Python: parse via None, then promote to np.nan in numeric columns.
        if 'anchor_ids' in df.columns and df['anchor_ids'].dtype == object:
            df['anchor_ids'] = df['anchor_ids'].apply(
                lambda s: ast.literal_eval(s.replace('nan', 'None'))
            )
        if 'ranges' in df.columns and df['ranges'].dtype == object:
            df['ranges'] = df['ranges'].apply(
                lambda s: [np.nan if v is None else v
                           for v in ast.literal_eval(s.replace('nan', 'None'))]
            )
        return df
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def get_anchors_positions(tech):
    anchors_file = os.path.join(DATA_DIR, 'reference', 'csv', 'anchor_coordinates.csv')
    anchors_all = pd.read_csv(anchors_file)
    tag = {'uwb': 'UWB', 'wifi': 'WIFI', 'ble': 'BLE'}.get(tech)
    if tag is None:
        raise ValueError(f"Unsupported technology: {tech}")
    sub = anchors_all[anchors_all['point_id'].str.contains(tag)]
    return sub.set_index('point_id')[['X_LOCAL', 'Y_LOCAL', 'Z_LOCAL']].T.to_dict('list')


def get_reference_positions(tech):
    ref_file = os.path.join(DATA_DIR, 'reference', 'csv', 'point_coordinates.csv')
    ref_df = pd.read_csv(ref_file)
    cols = {
        'ble':  ['X_LOCAL_BLE',  'Y_LOCAL_BLE',  'Z_LOCAL_BLE'],
        'uwb':  ['X_LOCAL_UWB2', 'Y_LOCAL_UWB2', 'Z_LOCAL_UWB2'],  # preprocess_uwb.py uses UWB2 only
        'wifi': ['X_LOCAL_WIFI', 'Y_LOCAL_WIFI', 'Z_LOCAL_WIFI'],
    }.get(tech)
    if cols is None:
        raise ValueError(f"Unsupported technology: {tech}")
    return ref_df.set_index('point_id')[cols].T.to_dict('list')


def load_hymn_data():
    """Return (df_ble, df_uwb, df_wifi) with anchor_ids normalized and remapped
    to the surveyed-position labels, ground-truth columns renamed to
    (ref_x, ref_y), and true_ranges / residual_ranges precomputed.
    """
    df_uwb = read_data('uwb')
    df_ble = read_data('ble')
    df_wifi = read_data('wifi')

    df_ble['anchor_ids'] = df_ble['anchor_ids'].apply(_remap_ble_ids)
    df_wifi['anchor_ids'] = df_wifi['anchor_ids'].apply(_remap_wifi_ids)

    anchor_pos_ble = get_anchors_positions('ble')
    df_ble['true_ranges'] = df_ble.apply(lambda row: compute_true_ranges(row, anchor_pos_ble, 'ble'), axis=1)
    df_ble['residual_ranges'] = df_ble.apply(compute_residual_ranges, axis=1)
    df_ble.rename(columns={'X_LOCAL_BLE': 'ref_x', 'Y_LOCAL_BLE': 'ref_y'}, inplace=True)

    anchor_pos_uwb = get_anchors_positions('uwb')
    df_uwb['true_ranges'] = df_uwb.apply(lambda row: compute_true_ranges(row, anchor_pos_uwb, 'uwb'), axis=1)
    df_uwb['residual_ranges'] = df_uwb.apply(compute_residual_ranges, axis=1)
    df_uwb.rename(columns={'X_LOCAL_UWB': 'ref_x', 'Y_LOCAL_UWB': 'ref_y'}, inplace=True)

    anchor_pos_wifi = get_anchors_positions('wifi')
    df_wifi['true_ranges'] = df_wifi.apply(lambda row: compute_true_ranges(row, anchor_pos_wifi, 'wifi'), axis=1)
    df_wifi['residual_ranges'] = df_wifi.apply(compute_residual_ranges, axis=1)
    df_wifi.rename(columns={'X_LOCAL_WIFI': 'ref_x', 'Y_LOCAL_WIFI': 'ref_y'}, inplace=True)

    return df_ble, df_uwb, df_wifi
