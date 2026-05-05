import numpy as np

from hymn.grid_calc import calculate


def compute_true_ranges(row, anchor_pos, tech):
    distances = []
    for anchor_id in row['anchor_ids']:
        if anchor_id in anchor_pos:
            tech_x, tech_y = anchor_pos[anchor_id][:2]
            dx = row['X_LOCAL_' + tech.upper()] - tech_x
            dy = row['Y_LOCAL_' + tech.upper()] - tech_y
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)
        else:
            distances.append(np.nan)
    return distances

def compute_residual_ranges(row):
    return list(np.array(row['true_ranges']) - np.array(row['ranges']))

def compute_bounds(anchors_ble, anchors_uwb, anchors_wifi):
    from hymn.io import get_reference_positions

    ref_positions_ble = get_reference_positions('ble')
    ref_positions_uwb = get_reference_positions('uwb')
    ref_positions_wifi = get_reference_positions('wifi')

    
    x_min_ref_ble = min([v[0] for v in ref_positions_ble.values()])
    x_max_ref_ble = max([v[0] for v in ref_positions_ble.values()])
    y_min_ref_ble = min([v[1] for v in ref_positions_ble.values()])
    y_max_ref_ble = max([v[1] for v in ref_positions_ble.values()])

    x_min_ref_uwb = min([v[0] for v in ref_positions_uwb.values()])
    x_max_ref_uwb = max([v[0] for v in ref_positions_uwb.values()])
    y_min_ref_uwb = min([v[1] for v in ref_positions_uwb.values()])
    y_max_ref_uwb = max([v[1] for v in ref_positions_uwb.values()])

    x_min_ref_wifi = min([v[0] for v in ref_positions_wifi.values()])
    x_max_ref_wifi = max([v[0] for v in ref_positions_wifi.values()])
    y_min_ref_wifi = min([v[1] for v in ref_positions_wifi.values()])
    y_max_ref_wifi = max([v[1] for v in ref_positions_wifi.values()])

    x_min_ref = min(x_min_ref_ble, x_min_ref_uwb, x_min_ref_wifi)
    x_max_ref = max(x_max_ref_ble, x_max_ref_uwb, x_max_ref_wifi)
    y_min_ref = min(y_min_ref_ble, y_min_ref_uwb, y_min_ref_wifi)
    y_max_ref = max(y_max_ref_ble, y_max_ref_uwb, y_max_ref_wifi)

    x_max_ble = np.array([a[0] for a in anchors_ble]).max()
    x_min_ble = np.array([a[0] for a in anchors_ble]).min()
    y_max_ble = np.array([a[1] for a in anchors_ble]).max()
    y_min_ble = np.array([a[1] for a in anchors_ble]).min()

    x_max_uwb = np.array([a[0] for a in anchors_uwb]).max()
    x_min_uwb = np.array([a[0] for a in anchors_uwb]).min()
    y_max_uwb = np.array([a[1] for a in anchors_uwb]).max()
    y_min_uwb = np.array([a[1] for a in anchors_uwb]).min()

    x_max_wifi = np.array([a[0] for a in anchors_wifi]).max()
    x_min_wifi = np.array([a[0] for a in anchors_wifi]).min()
    y_max_wifi = np.array([a[1] for a in anchors_wifi]).max()
    y_min_wifi = np.array([a[1] for a in anchors_wifi]).min()

    x_min = min(
        np.array([x_max_uwb, x_min_uwb, x_max_ble, x_min_ble, x_max_wifi, x_min_wifi, x_min_ref, x_max_ref]))
    x_max = max(
        np.array([x_max_uwb, x_min_uwb, x_max_ble, x_min_ble, x_max_wifi, x_min_wifi, x_min_ref, x_max_ref]))
    y_min = min(
        np.array([y_max_uwb, y_min_uwb, y_max_ble, y_min_ble, y_max_wifi, y_min_wifi, y_min_ref, y_max_ref]))
    y_max = max(
        np.array([y_max_uwb, y_min_uwb, y_max_ble, y_min_ble, y_max_wifi, y_min_wifi, y_min_ref, y_max_ref]))

    max_all = float(max(np.array([x_min, x_max, y_min, y_max])))
    min_all = float(min(np.array([x_min, x_max, y_min, y_max])))
    return min_all, max_all

def compute_residual_maps(df, grid_range, anchor_pos, tech, grid_res=0.1):
    min_all, max_all = grid_range
    return calculate(df.copy(), anchor_pos, grid_res=grid_res,
                     grid_range_x=(min_all, max_all),
                     grid_range_y=(min_all, max_all), tech=tech)