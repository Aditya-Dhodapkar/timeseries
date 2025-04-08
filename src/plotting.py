import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


plt.style.use('default')
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def smooth(y, box_pts=1):
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
    
    # Make sure y is a 1D array
    if len(y.shape) > 1:
        y = y.flatten()
        
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os, torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def smooth(y, box_pts=1):
    # Helper function to smooth the curve if needed
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
    if len(y.shape) > 1:
        y = y.flatten()
    box = np.ones(box_pts)/box_pts
    return np.convolve(y, box, mode='same')

# def plotter(name, y_true, y_pred, ascore, labels):
#     # Convert tensors to numpy and reshape predictions if needed
#     if isinstance(y_true, torch.Tensor):
#         y_true = y_true.cpu().numpy()
#     if isinstance(y_pred, (torch.Tensor, np.ndarray)):
#         if y_pred.shape[0] == 8760:  # hourly to daily
#             y_pred = y_pred.reshape(365, 24, -1).mean(axis=1)

#     print("Verification in plotter:")
#     print("First value of true data:", y_true[0,0])
#     print("First value of predicted data:", y_pred[0,0])
    
#     if isinstance(ascore, torch.Tensor):
#         ascore = ascore.cpu().numpy()
#     if ascore.shape[0] == 8760:  # Need to reshape ascore too!
#         ascore = ascore.reshape(365, 24, -1).mean(axis=1)
    
#     # If y_true is still 3D, average over the second dimension (e.g., hours)
#     if len(y_true.shape) == 3:
#         y_true = y_true.mean(axis=1)
   
#     print(f"Plotter shapes after processing:")
#     print(f"y_true: {y_true.shape}")
#     print(f"y_pred: {y_pred.shape}")
#     print(f"ascore: {ascore.shape}")

#     # Optional shift for TranAD (if needed)
#     if 'TranAD' in name:
#         y_true = np.roll(y_true, 1, 0)
       
#     os.makedirs(os.path.join('plots', name), exist_ok=True)
#     pdf = PdfPages(f'plots/{name}/output.pdf')
   
#     # Calculate metrics for each dimension
#     metrics_text = []
#     for dim in range(y_true.shape[1]):
#         y_t, y_p = y_true[:, dim], y_pred[:, dim]
#         mse = mean_squared_error(y_t, y_p)
#         rmse = np.sqrt(mse)
#         mae = mean_absolute_error(y_t, y_p)
#         r2 = r2_score(y_t, y_p)
#         metrics_text.append(f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}')

#     # Plot each dimension in a separate 2-subplot figure
#     for dim in range(y_true.shape[1]):
#         y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
       
#         # Removed sharex=True so the top subplot will show its x-axis
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
#         plt.rcParams.update({'font.size': 8})  # Adjust font size if desired
       
#         ax1.set_xlabel('Time Step (Days)')
#         ax1.set_ylabel('Normalized Value')

#         if dim < 3:
#             title = f'Temporal Feature {dim}'
#         elif dim < 8:
#             title = f'Temperature Site {dim-2}'
#         else:
#             title = f'GHI Site {dim-7}'
#         ax1.set_title(title, fontsize=10)

#         x_axis = np.arange(365) 
#         ax1.plot(x_axis, smooth(y_t), linewidth=0.8, label='True')
#         ax1.plot(x_axis, smooth(y_p), '-', alpha=0.6, linewidth=0.8, label='Predicted')
        
#         # Always show legend
#         ax1.legend(loc='upper right', fontsize=8)

#         # Add metrics text box
#         ax1.text(
#             0.02, 0.98, metrics_text[dim],  
#             transform=ax1.transAxes,
#             verticalalignment='top',
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
#             fontsize=8
#         )
       
#         # Overlay the labels on a secondary y-axis
#         ax3 = ax1.twinx()
#         ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
#         ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
       
#         # Bottom subplot: anomaly score
#         ax2.plot(x_axis, smooth(a_s), linewidth=0.2, color='g')
#         ax2.set_xlim(0, 365)
#         ax2.set_xlabel('Time Step (Days)')
#         ax2.set_ylabel('Anomaly Score')
       
#         pdf.savefig(fig)
#         plt.close()
   
#     pdf.close()
    
# def plot_daily_patterns(true_data, pred_data, name):
#    print("Initial shapes:")
#    print(f"true_data shape: {true_data.shape}")
#    print(f"pred_data shape: {pred_data.shape}")

#    # If pred_data is hourly (8760), reshape to daily (365)
#    if pred_data.shape[0] == 8760:
#        pred_data = pred_data.reshape(365, 24, -1).mean(axis=1)
#        print(f"After reshaping pred_data shape: {pred_data.shape}")

#    # If true_data is 3D, take mean over hours
#    if len(true_data.shape) == 3:
#        true_data = true_data.mean(axis=1)
#        print(f"After reshaping true_data shape: {true_data.shape}")

#    print("Final shapes before metrics:")
#    print(f"true_data shape: {true_data.shape}")
#    print(f"pred_data shape: {pred_data.shape}")

#    plt.figure(figsize=(15, 8))
#    plt.rcParams.update({'font.size': 8})
   
#    # Calculate metrics on the 2D data
#    mse = mean_squared_error(true_data, pred_data)
#    rmse = np.sqrt(mse)
#    mae = mean_absolute_error(true_data, pred_data)
#    r2 = r2_score(true_data.flatten(), pred_data.flatten())
   
#    metric_text = f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
   
#    # Plot for each feature
#    n_features = min(true_data.shape[1], pred_data.shape[1])
   
#    for i in range(n_features):
#        plt.subplot(3, 5, i+1)
#        plt.plot(true_data[:, i], label='True', alpha=0.7)
#        plt.plot(pred_data[:, i], label='Predicted', alpha=0.7)
       
#        if i < 3:
#            plt.title(f'Temporal Feature {i}')
#        elif i < 8:
#            plt.title(f'Temperature Site {i-2}')
#        else:
#            plt.title(f'GHI Site {i-7}')
           
#        plt.xlabel('Days')
#        plt.ylabel('Normalized Value')
           
#        if i == 0:
#            plt.legend()
#            plt.text(0.02, 0.98, metric_text, 
#                    transform=plt.gca().transAxes,
#                    verticalalignment='top',
#                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
#                    fontsize=8)
   
#    plt.tight_layout()
#    plt.savefig(f'plots/{name}_daily_patterns.png')
#    plt.close()


# def plot_training_curve(accuracy_list, name):
#     losses = [x[0] for x in accuracy_list]
#     plt.figure(figsize=(10, 6))
#     plt.rcParams.update({'font.size': 8})
#     plt.plot(losses, label='Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('MSE Loss')
#     plt.title(f'Training Loss Over Time - {name}', fontsize=10)
#     plt.legend(fontsize=8)
#     plt.savefig(f'plots/{name}_training_loss.png')
#     plt.close()

from scipy.ndimage.filters import uniform_filter1d

def plot_load_predictions(y_true, y_pred, name):
    mse, rmse, mae, r2 = calculate_metrics(y_true, y_pred)
    metric_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}"

    # Optionally smooth the curves
    window_size = 24  # e.g., average over one day if hourly
    y_true_smooth = uniform_filter1d(y_true, size=window_size)
    y_pred_smooth = uniform_filter1d(y_pred, size=window_size)

    plt.figure(figsize=(15, 6))
    plt.plot(y_true_smooth, label="True Load (smoothed)", alpha=0.7)
    plt.plot(y_pred_smooth, label="Predicted Load (smoothed)", alpha=0.7)
    plt.xlabel("Time Step (Hourly)")
    plt.ylabel("Normalized Load")
    plt.title("Load Predictions (Smoothed)")
    plt.legend()
    plt.text(0.02, 0.98, metric_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_load_predictions_smoothed.png")
    plt.close()

# def plot_load_by_month(y_true_daily, y_pred_daily, name):
#     """
#     Plot the true vs. reconstructed daily load averages for each month separately.
#     Print MSE, RMSE, MAE, and R² in the top-left corner of each graph.
#     Assumes y_true_daily and y_pred_daily are 1D arrays of length ~365 (non-leap year).
#     """
#     from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
#     months = {
#         "Jan": (0, 31),
#         "Feb": (31, 59),
#         "Mar": (59, 90),
#         "Apr": (90, 120),
#         "May": (120, 151),
#         "Jun": (151, 181),
#         "Jul": (181, 212),
#         "Aug": (212, 243),
#         "Sep": (243, 273),
#         "Oct": (273, 304),
#         "Nov": (304, 334),
#         "Dec": (334, 365)
#     }
    
#     for month, (start, end) in months.items():
#         y_true_month = y_true_daily[start:end]
#         y_pred_month = y_pred_daily[start:end]
        
#         if len(y_true_month) == 0:
#             continue
        
#         mse = mean_squared_error(y_true_month, y_pred_month)
#         rmse = mse**0.5
#         mae = mean_absolute_error(y_true_month, y_pred_month)
#         r2 = r2_score(y_true_month, y_pred_month)
        
#         metric_text = (f"MSE: {mse:.4f}\n"
#                        f"RMSE: {rmse:.4f}\n"
#                        f"MAE: {mae:.4f}\n"
#                        f"R²: {r2:.4f}")
        
#         plt.figure(figsize=(10, 5))
#         x_axis = np.arange(start, end)
#         plt.plot(x_axis, y_true_month, label="True Daily Average", alpha=0.7, linewidth=1.5)
#         plt.plot(x_axis, y_pred_month, label="Reconstructed Daily Average", alpha=0.7, linewidth=1.5)
#         plt.xlabel("Day of Year")
#         plt.ylabel("Load")
#         plt.title(f"Load reconstruction for {month}")
#         plt.legend()
        
        
#         plt.text(0.02, 0.98, metric_text, transform=plt.gca().transAxes,
#                  verticalalignment='top', fontsize=9,
#                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
#         plt.tight_layout()
#         plt.savefig(f"plots/{name}_{month}_load_predictions.png")
#         plt.close()


def plot_three_lines_by_month(year1_daily, year2_daily, year3_pred, name):
    """
    Plot 3 lines for each month:
      - Year 1 daily load
      - Year 2 daily load
      - Predicted daily load for Year 3
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    months = {
        "Jan": (0, 31),
        "Feb": (31, 59),
        "Mar": (59, 90),
        "Apr": (90, 120),
        "May": (120, 151),
        "Jun": (151, 181),
        "Jul": (181, 212),
        "Aug": (212, 243),
        "Sep": (243, 273),
        "Oct": (273, 304),
        "Nov": (304, 334),
        "Dec": (334, 365)
    }

    for month, (start, end) in months.items():
        y1 = year1_daily[start:end]
        y2 = year2_daily[start:end]
        y3p = year3_pred[start:end]
        
        if len(y1) == 0 or len(y2) == 0 or len(y3p) == 0:
            continue

        # For demonstration, let's just measure error vs. y3p (assuming that's the main interest).
        # Or if you want error vs. year2, etc., adjust accordingly.
        mse = mean_squared_error(y2, y3p)
        rmse = mse**0.5
        mae = mean_absolute_error(y2, y3p)
        r2 = r2_score(y2, y3p)

        metric_text = (f"MSE: {mse:.4f}\n"
                       f"RMSE: {rmse:.4f}\n"
                       f"MAE: {mae:.4f}\n"
                       f"R²: {r2:.4f}")

        plt.figure(figsize=(10, 5))
        x_axis = np.arange(start, end)
        
        # Plot all three lines
        plt.plot(x_axis, y1, label="Year 1 (Training)", alpha=0.7, linewidth=1.5)
        plt.plot(x_axis, y2, label="Year 2 (True)", alpha=0.7, linewidth=1.5)
        plt.plot(x_axis, y3p, label="Year 3 (Predicted)", alpha=0.7, linewidth=1.5)

        plt.xlabel("Day of Year")
        plt.ylabel("Load")
        plt.title(f"Load Comparison for {month}")
        plt.legend()

        plt.text(0.02, 0.98, metric_text, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"plots/{name}_{month}_three_lines.png")
        plt.close()