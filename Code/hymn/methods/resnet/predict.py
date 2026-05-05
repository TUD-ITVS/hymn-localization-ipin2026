import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import euclidean

from hymn.config import RESNET, PREDICTIONS_DIR
from hymn.grid_calc import calculate
from hymn.preprocessing import compute_bounds


def model_predict_test(path, X, y):
    model = tf.keras.models.load_model(path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=RESNET["learning_rate"], clipnorm=1.0),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    predictions = model.predict(X)

    df = pd.DataFrame({'actual': [(x, y) for x, y in y],
                       'predicted': [(x, y) for x, y in predictions]})

    # df['invalid_mess'] = [
    #     np.array([(maps[0].sum() == 0) for maps in ranges]).sum()/2 for ranges in X]
    df['error(m)'] = df.apply(lambda row: euclidean(row['actual'], row['predicted']), axis=1)

    return df

def model_predict_csv(model_path, test_data_path, anchors_ble, anchors_uwb, anchors_wifi, tech):
    # 1. Setup
    anchors_ble_arr = anchors_ble
    anchors_uwb_arr = anchors_uwb
    anchors_wifi_arr = anchors_wifi
    bounds = compute_bounds(anchors_ble_arr, anchors_uwb_arr, anchors_wifi_arr)
    calc_args = {'grid_res': 0.5, 'grid_range_x': bounds, 'grid_range_y': bounds}
    
    df = pd.read_pickle(test_data_path).reset_index(drop=True)
    
    # 2. Dynamic Calculation (Initialize both as None)
    df_ble, df_uwb, df_wifi = None, None, None
    X_list = []

    # Calculate BLE if tech is 'ble' or 'both'
    if tech in ['ble', 'both']:
        # If 'both', slice first 8 cols, else use whole df
        data = df.iloc[:, :8] if tech == 'both' else df
        df_ble = calculate(data, anchors_ble_arr, **calc_args, tech='ble')
        X_list.append(np.array(df_ble['residual_range_map'].tolist()))

    # Calculate UWB if tech is 'uwb' or 'both'
    if tech in ['uwb', 'both']:
        # If 'both', slice cols after 8, else use whole df
        data = df.iloc[:, 8:] if tech == 'both' else df
        df_uwb = calculate(data, anchors_uwb_arr, **calc_args, tech='uwb')
        X_list.append(np.array(df_uwb['residual_range_map'].tolist()))

    # Calculate WiFi if tech is 'wifi' or 'both'
    if tech in ['wifi', 'both']:
        # If 'both', slice cols after 8, else use whole df
        data = df.iloc[:, 8:] if tech == 'both' else df
        df_wifi = calculate(data, anchors_wifi_arr, **calc_args, tech='wifi')
        X_list.append(np.array(df_wifi['residual_range_map'].tolist()))

    # 3. Prepare Inputs
    # Concatenate features if both exist (axis 1), then transpose
    X = np.concatenate(X_list, axis=1).transpose(0, 2, 3, 1)
    
    # Use BLE as reference for XY/MessID, fallback to UWB if BLE is missing
    ref_df = df_ble if df_ble is not None else df_uwb if df_uwb is not None else df_wifi
    sample_xy = ref_df[['ref_x', 'ref_y']].to_numpy(dtype=np.float32)

    # 4. Predict
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=RESNET["learning_rate"], clipnorm=1.0),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    pred = model.predict(X)

    # 5. Build Output
    out = ref_df[['point_id']].copy()
    out['timestamp_ble'] = df_ble['ts'] if df_ble is not None else None
    out['timestamp_uwb'] = df_uwb['ts'] if df_uwb is not None else None
    out['timestamp_wifi'] = df_wifi['ts'] if df_wifi is not None else None
    out['technology'] = tech.upper()
    out[['ref_x', 'ref_y']] = sample_xy
    
    out['n_anchors_ble'] = df_ble['ranges'].map(len) if df_ble is not None else 0
    out['n_anchors_uwb'] = df_uwb['ranges'].map(len) if df_uwb is not None else 0
    out['n_anchors_wifi'] = df_wifi['ranges'].map(len) if df_wifi is not None else 0
    
    out['x_pred'], out['y_pred'] = pred[:, 0], pred[:, 1]
    
    # Vectorized Euclidean distance (much faster than apply/lambda)
    out['error(m)'] = np.linalg.norm(sample_xy - pred, axis=1)

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out.to_csv(os.path.join(PREDICTIONS_DIR, f"{tech}_predictions.csv"), index=False)
    out.to_pickle(os.path.join(PREDICTIONS_DIR, f"{tech}_predictions.pkl"))

    return out