import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hymn.config import TRAINED_DIR
from hymn.io import get_anchors_positions

plt.rcParams.update({"font.family": "serif",
                     "font.serif": "Times New Roman",
                     'font.size': 14,
                    #  'axes.labelsize': 16,
                     'xtick.labelsize': 12,
                     'ytick.labelsize': 12,
                     'legend.fontsize': 11,
                     })

def visualize_train_sample(X_sample, y_sample, bounds, tech):
    anchors_ble_arr = np.array(list(get_anchors_positions('ble').values()))
    anchors_uwb_arr = np.array(list(get_anchors_positions('uwb').values()))
    anchors_wifi_arr = np.array(list(get_anchors_positions('wifi').values()))

    
    if tech == 'ble':
        resmap_plot(X_sample, bounds, 'residual_range_map', only_array = True, y = y_sample, anchors_ble = anchors_ble_arr)
    elif tech == 'uwb':
        resmap_plot(X_sample, bounds, 'residual_range_map', only_array = True, y = y_sample, anchors_uwb = anchors_uwb_arr)
    elif tech == 'wifi':
        resmap_plot(X_sample, bounds, 'residual_range_map', only_array = True, y = y_sample, anchors_wifi = anchors_wifi_arr)
    else:
        resmap_plot(X_sample, bounds, 'residual_range_map', only_array = True, y = y_sample, anchors_uwb = anchors_uwb_arr, anchors_ble = anchors_ble_arr, anchors_wifi = anchors_wifi_arr)
  
def resmap_plot(sample, extent, column, only_array = False, y = (0,0), anchors_ble = None, anchors_uwb = None, anchors_wifi = None):
    fig, ax = plt.subplots(figsize=(6, 6), layout = 'constrained')  # Create figure and axis
    
    if only_array:
        for residual_map in sample:
            im = ax.imshow(residual_map, 
                            extent=extent, 
                           origin='lower', cmap='viridis', alpha=0.45)
    
    else:
        # Plotting all the residual maps for a single Tag location
        for residual_map in sample[column]:
            im = ax.imshow(residual_map, 
                            extent=extent, 
                           origin='lower', cmap='viridis', alpha=0.45)
    
    # Create colorbar with proper size
    cbar = fig.colorbar(im, ax=ax, shrink=0.45, aspect=10)  # Adjust shrink & aspect
    cbar.set_label('Likelihood Values')

    # Plot anchors and tag position
    if anchors_ble is not None:
        ax.scatter([pos[0] for pos in anchors_ble], [pos[1] for pos in anchors_ble], marker='o', color='crimson', label='Anchor BLE')
    if anchors_uwb is not None:
        ax.scatter([pos[0] for pos in anchors_uwb], [pos[1] for pos in anchors_uwb], marker='s', color='darkorange', label='Anchor UWB')
    if anchors_wifi is not None:
        ax.scatter([pos[0] for pos in anchors_wifi], [pos[1] for pos in anchors_wifi], marker='^', color='blue', label='Anchor WiFi')
    
    if only_array:
        ax.scatter(y[0], y[1], color='white', label='Reference position', marker = 'x')
    
    else:
        ax.scatter(sample['ref_x'], sample['ref_y'], color='white', label='Reference position', marker = 'x')
    
    ax.legend(loc = 'upper left')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    # ax.set_ylim([-1, 12.5])
    # ax.set_xlim([3, 11])
    # ax.set_title('Likelihood Grid Map of '+ measurement)
    
    plt.show()


def plot_training_history(history):
    # Plots training and validation loss over epochs 
    plt.figure(figsize=(5, 4), layout = 'constrained')
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    # plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (m)')
    plt.legend(loc='upper right')
    plt.grid(alpha = 0.25, linestyle = '--')
    
    ts = int(datetime.timestamp(datetime.now()))
    os.makedirs(TRAINED_DIR, exist_ok=True)
    plt.savefig(os.path.join(TRAINED_DIR, f'loss_over_epochs{ts}.pdf'), bbox_inches='tight', format='pdf', dpi=300)
    plt.show()

def ecdf_stats(ecdf_x):
    # Find percentiles
    percentiles = [0.25, 0.5, 0.75, 0.95]
    percentile_values = np.quantile(ecdf_x, percentiles)  # Get exact percentiles
    
    col_labels = ['Percentile (%)','Error (m)']
    table_vals = [[round(percentiles[0]*100), round(percentile_values[0], 3)],
                  [round(percentiles[1]*100), round(percentile_values[1], 3)],
                  [round(percentiles[2]*100), round(percentile_values[2], 3)],
                  [round(percentiles[3]*100), round(percentile_values[3], 3)],
                 ]
    
    table = plt.table(cellText = table_vals,
              colWidths = [0.1]*3,
              colLabels=col_labels,
              loc='upper right', bbox=[0.58, 0.78, 0.39, 0.20])
    
    table.auto_set_font_size(False)

    for key, cell in table.get_celld().items():
        cell.get_text().set_fontsize(9)    # Increase font size
        cell.set_linewidth(0.1) 
        
def plot_ecdf(ecdf_x, error, params, save = True):
    plt.figure(figsize = (5,4), layout = 'constrained')
    
    sns.ecdfplot(error)
    plt.xlabel('Positioning error (m)')
    plt.ylabel('ECDF')
    plt.grid(alpha = 0.25, linestyle = '--')
    
    ecdf_stats(ecdf_x)
    
    params_str = "-".join(map(str, params))
    
    if save == True:
        ts = int(datetime.timestamp(datetime.now()))
        os.makedirs(TRAINED_DIR, exist_ok=True)
        plt.savefig(os.path.join(TRAINED_DIR, f'ecdf_plot_{params_str}_{ts}.pdf'), bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
    
def predict_plot(sample, extent, predicted, sample_xy=None):
    fig, ax = plt.subplots(figsize=(8, 6), layout = 'constrained')  # Create figure and axis
    
    for residual_map in sample:
        im = ax.imshow(residual_map, 
                        extent= extent, 
                       origin='lower', cmap='viridis', alpha=0.15)
        

    # Create colorbar with proper size
    cbar = fig.colorbar(im, ax=ax, shrink=0.45, aspect=10)  # Adjust shrink & aspect
    cbar.set_label('Likelihood Values')

    # Plot anchors and tag position
    # ax.scatter([pos[0] for pos in anchors], [pos[1] for pos in anchors], color='red', label='anchors')
    if sample_xy is not None:
        ax.scatter(sample_xy[0], sample_xy[1], color='white', label='Reference position')
    
    ax.scatter(predicted[0], predicted[1], marker = '*', color = 'k', label = 'Predicted position')
        
    ax.legend()
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    # ax.set_ylim([-5, 9])
    # ax.set_title('Predictions')
    
    plt.show()