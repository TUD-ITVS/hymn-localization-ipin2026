"""ResNet training entry point.

Usage (from Code/):
    python scripts/train_resnet.py
or
    python -m scripts.train_resnet
"""
import json
import os
import sys
import warnings
from datetime import datetime

# Ensure Code/ is on sys.path so `import hymn` works regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from hymn.config import TRAINED_DIR, TRAINING_SUMMARIES_DIR
from hymn.io import get_anchors_positions, load_hymn_data
from hymn.preprocessing import compute_bounds, compute_residual_maps
from hymn.methods.resnet.data_split import split_leave_points_out, split_random
from hymn.methods.resnet.train import model_training
from hymn.methods.resnet.predict import model_predict_test, model_predict_csv
from hymn.plotting.diagnostics import plot_ecdf, visualize_train_sample

warnings.filterwarnings("ignore")


def run_scenario(technology, only_predict=False, predict_dict=None, use_aug=False, use_random_split=False):
    df_ble, df_uwb, df_wifi = load_hymn_data()

    anchors_ble_arr = np.array(list(get_anchors_positions('ble').values()))
    anchors_uwb_arr = np.array(list(get_anchors_positions('uwb').values()))
    anchors_wifi_arr = np.array(list(get_anchors_positions('wifi').values()))

    min_all, max_all = compute_bounds(anchors_ble_arr, anchors_uwb_arr, anchors_wifi_arr)

    data_ble = compute_residual_maps(df_ble, (min_all, max_all), anchors_ble_arr, 'ble', grid_res=0.5)
    data_uwb = compute_residual_maps(df_uwb, (min_all, max_all), anchors_uwb_arr, 'uwb', grid_res=0.5)
    data_wifi = compute_residual_maps(df_wifi, (min_all, max_all), anchors_wifi_arr, 'wifi', grid_res=0.5)

    if technology.lower() == 'ble':
        ble_in, uwb_in, wifi_in = data_ble, None, None
    elif technology.lower() == 'uwb':
        ble_in, uwb_in, wifi_in = None, data_uwb, None
    elif technology.lower() == 'wifi':
        ble_in, uwb_in, wifi_in = None, None, data_wifi
    elif technology.lower() in ('both', 'combo', 'combined'):
        ble_in, uwb_in, wifi_in = data_ble, data_uwb, data_wifi
    else:
        raise ValueError("technology must be one of ['ble','uwb','wifi','both']")

    if use_random_split:
        print('Random-split protocol: partitioning epochs uniformly at random across all reference points...')
        X_train, X_test, y_train, y_test = split_random(ble_in, uwb_in, wifi_in, transpose=True, aug=use_aug)
    else:
        print('Spatial-holdout protocol: holding entire reference points out of training (leave-points-out)...')
        X_train, X_test, y_train, y_test = split_leave_points_out(ble_in, uwb_in, wifi_in, transpose=True, aug=use_aug)

    if only_predict:
        print('Only prediction requested, skipping training...')
        best_model_path = predict_dict['model_path']
        data_path = predict_dict['data_path']
        return model_predict_csv(best_model_path, data_path, anchors_ble_arr, anchors_uwb_arr, anchors_wifi_arr, technology.lower())

    print(f'X_Train: {len(X_train)}, X_test: {len(X_test)}')

    shape = X_train[0].shape
    history, hist_json_file = model_training(X_train, X_test, y_train, y_test, shape, (technology, use_aug, use_random_split))

    # Evaluate with the latest .h5 saved by the checkpoint callback
    trained_models = [f for f in os.listdir(TRAINED_DIR) if f.endswith('.h5')]
    if not trained_models:
        raise FileNotFoundError(f'No trained model .h5 found in {TRAINED_DIR}')
    last_model = sorted(trained_models, key=lambda x: os.path.getmtime(os.path.join(TRAINED_DIR, x)))[-1]
    model_path = os.path.join(TRAINED_DIR, last_model)

    test_predicted = model_predict_test(model_path, X_test, y_test)
    errors = test_predicted['error(m)'].values
    ecdf = sm.distributions.ECDF(errors)

    metrics = {
        'technology': technology.lower(),
        'test_count': int(len(errors)),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'p25': float(np.quantile(errors, 0.25)),
        'p75': float(np.quantile(errors, 0.75)),
        'p95': float(np.quantile(errors, 0.95)),
        'min_all': min_all,
        'max_all': max_all,
        'use_aug': use_aug,
        'split_kind': 'random' if use_random_split else 'spatial',
        'model_path': model_path,
        'history_path': hist_json_file,
        'ecdf_errors': ([json.dumps(ecdf.x.tolist())], [json.dumps(errors.tolist())])
    }

    plot_ecdf(ecdf.x, errors, (technology, use_aug, use_random_split))
    return metrics


def main():
    scenarios = ['both']
    protocols = [False, True]  # spatial-holdout first, then random-split

    for use_random_split in protocols:
        split_kind = 'random' if use_random_split else 'spatial'
        rows = []
        for scenario in scenarios:
            print(f'\n=== Training scenario={scenario}, protocol={split_kind} ===')
            metrics = run_scenario(technology=scenario, only_predict=False, use_aug=False, use_random_split=use_random_split)
            rows.append(metrics)

        df_results = pd.DataFrame(rows)
        os.makedirs(TRAINING_SUMMARIES_DIR, exist_ok=True)
        ts = int(datetime.timestamp(datetime.now()))
        df_results.to_csv(os.path.join(TRAINING_SUMMARIES_DIR, f'results_split-{split_kind}-{ts}.csv'), index=False)


if __name__ == "__main__":
    main()
