import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import torch
from src.folderconstants import output_folder
import matplotlib.pyplot as plt

def normalize3(a, min_a=None, max_a=None):
    if min_a is None:
        min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def load_training_data(dataset):
    dataset_folder = os.path.join("data", dataset)
    df = pd.read_excel(os.path.join(dataset_folder, "training.xlsx"))
    print(f"Reading training data from: {os.path.join(dataset_folder, 'training.xlsx')}")
    print(f"DataFrame shape: {df.shape}")
    # For this advanced regression task, we use Years 1 and 2.
    train_df = df[df['Year'].isin([1,2])]
    print(f"Using training data for Year 1 and 2: {train_df.shape}")
    return train_df

def load_testing_data(dataset):
    dataset_folder = os.path.join("data", dataset)
    df = pd.read_excel(os.path.join(dataset_folder, "testing.xlsx"))
    print(f"Reading testing data from: {os.path.join(dataset_folder, 'testing.xlsx')}")
    print(f"DataFrame shape: {df.shape}")
    return df

def preprocess_with_autoencoder(dataset, autoencoder_model, window_length=24, temp_autoencoder_model=None, ghi_autoencoder_model=None):
    """
    Preprocesses data for the hourly autoencoder+regression pipeline with cross‑validation and advanced feature extraction.
    
    For training (Years 1 & 2 from training.xlsx):
      - Extract static features (temporal, temperature, GHI) → 13 columns.
      - Create a unique day identifier using Year, Month, and Day.
      - Split training data (by unique day_id) into training and validation sets.
      - For each complete day (with exactly window_length hours), extract:
            • Load values (shape: (window_length,))
            • In advanced mode: Temperature and GHI values (transposed so each site becomes a row).
      - Compute latent representations for the load using the autoencoder’s encoder (process daywise).
      - If advanced mode is active, also compute latent representations for temperature and GHI using their respective autoencoders.
      - Expand each day’s latent vector by repeating it window_length times.
      - In advanced mode, combine the load latent with averaged temperature and GHI latent features (across 5 sites) to form a combined latent vector.
      - Append seasonal features (sin/cos of day-of-year computed from Month/Day).
      - Concatenate the static features (now with added seasonal features) with the combined latent features to yield, for example, a 29‑dimensional feature vector per hourly sample (13 static + 16 latent if advanced mode is used).
      - Save the combined hourly features and the original hourly targets.
    
    For testing (Year 3 from testing.xlsx):
      - Process static features similarly.
      - Since the 'Load' column is empty, create a dummy load array (zeros), reshape using window_length, and extract latent features (which will be zeros).
      - Concatenate with static features and save.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    
    # Determine if advanced mode is used
    use_advanced = (temp_autoencoder_model is not None and ghi_autoencoder_model is not None)
    if use_advanced:
        print("Using advanced feature extraction with temperature and GHI autoencoders")
    
    # Load DataFrames
    train_df = load_training_data(dataset)
    test_df = load_testing_data(dataset)
    
    # Define columns
    temporal_cols = ['Month', 'Day', 'Hour']
    temp_cols = ['Site-1 Temp', 'Site-2 Temp', 'Site-3 Temp', 'Site-4 Temp', 'Site-5 Temp']
    ghi_cols = ['Site-1 GHI', 'Site-2 GHI', 'Site-3 GHI', 'Site-4 GHI', 'Site-5 GHI']
    feature_cols = temporal_cols + temp_cols + ghi_cols  # 13 columns

    # Create unique day_id
    train_df['day_id'] = train_df['Year'] * 10000 + train_df['Month'] * 100 + train_df['Day']
    
    # Split days for cross-validation (80% train, 20% validation)
    unique_day_ids = train_df['day_id'].unique()
    np.random.seed(42)
    shuffled_day_ids = np.random.permutation(unique_day_ids)
    split_idx = int(0.8 * len(shuffled_day_ids))
    train_day_ids = shuffled_day_ids[:split_idx]
    val_day_ids = shuffled_day_ids[split_idx:]
    
    actual_train_df = train_df[train_df['day_id'].isin(train_day_ids)]
    val_df = train_df[train_df['day_id'].isin(val_day_ids)]
    
    print(f"Train set: {actual_train_df.shape[0]} hours; Validation set: {val_df.shape[0]} hours")
    
    # Process static features (hourly) for all sets
    train_static = actual_train_df[feature_cols].values   # (num_hours_train, 13)
    val_static = val_df[feature_cols].values              # (num_hours_val, 13)
    test_static = test_df[feature_cols].values            # (num_hours_test, 13)
    
    # Extract hourly targets
    train_targets = actual_train_df['Load'].values        # (num_hours_train,)
    val_targets = val_df['Load'].values                   # (num_hours_val,)
    test_targets = np.zeros(test_df.shape[0])             # (num_hours_test,)
    
    # Group data by complete day (only include days with exactly window_length hours)
    def group_by_day(df):
        grouped = []
        for day_id in np.unique(df['day_id']):
            day_data = df[df['day_id'] == day_id].sort_values('Hour')
            if len(day_data) == window_length:
                grouped.append(day_data)
        return grouped
    
    train_days = group_by_day(actual_train_df)
    val_days = group_by_day(val_df)
    
    # Build lists for daywise load values and, if advanced, for temperature and GHI
    train_load_days = []
    val_load_days = []
    if use_advanced:
        train_temp_days = []
        train_ghi_days = []
        val_temp_days = []
        val_ghi_days = []
    
    for day_data in train_days:
        train_load_days.append(day_data['Load'].values)  # shape: (window_length,)
        if use_advanced:
            train_temp_days.append(day_data[temp_cols].values.T)  # (5, window_length)
            train_ghi_days.append(day_data[ghi_cols].values.T)     # (5, window_length)
            
    for day_data in val_days:
        val_load_days.append(day_data['Load'].values)
        if use_advanced:
            val_temp_days.append(day_data[temp_cols].values.T)
            val_ghi_days.append(day_data[ghi_cols].values.T)
    
    # Convert daywise arrays
    train_loads_array = np.array(train_load_days)   # (num_train_days, window_length)
    val_loads_array = np.array(val_load_days)         # (num_val_days, window_length)
    
    # Use autoencoder to obtain latent features from load data
    train_loads_tensor = torch.from_numpy(train_loads_array).float().to(device)
    autoencoder_model.eval()
    with torch.no_grad():
        train_latent = autoencoder_model.encoder(train_loads_tensor).cpu().numpy()  # (num_train_days, latent_dim)
    
    val_loads_tensor = torch.from_numpy(val_loads_array).float().to(device)
    with torch.no_grad():
        val_latent = autoencoder_model.encoder(val_loads_tensor).cpu().numpy()  # (num_val_days, latent_dim)
    
    # Expand latent features to hourly level by repeating each vector window_length times
    train_latent_expanded = np.repeat(train_latent, window_length, axis=0)  # (num_train_days*window_length, latent_dim)
    val_latent_expanded = np.repeat(val_latent, window_length, axis=0)
    
    # For testing, use zeros for load values
    test_days_count = test_df.shape[0] // window_length
    test_loads_dummy = np.zeros((test_days_count, window_length))
    test_loads_tensor = torch.from_numpy(test_loads_dummy).float().to(device)
    with torch.no_grad():
        test_latent = autoencoder_model.encoder(test_loads_tensor).cpu().numpy()  # (num_test_days, latent_dim)
    test_latent_expanded = np.repeat(test_latent, window_length, axis=0)
    
    # Advanced: If using temp and GHI autoencoders, compute their latent features
    if use_advanced:
        # Reshape temperature and GHI data (average across sites later)
        train_temps = np.array([d[temp_cols].values.T for d in train_days])  # shape: (num_train_days, 5, window_length)
        train_ghis = np.array([d[ghi_cols].values.T for d in train_days])
        val_temps = np.array([d[temp_cols].values.T for d in val_days])
        val_ghis = np.array([d[ghi_cols].values.T for d in val_days])
        
        # Dummy zeros for test temperature and GHI (if not available)
        test_temps = np.zeros((test_days_count, 5, window_length))
        test_ghis = np.zeros((test_days_count, 5, window_length))
        
        temp_autoencoder_model.eval()
        ghi_autoencoder_model.eval()
        train_temps_tensor = torch.from_numpy(train_temps.reshape(-1, window_length)).float().to(device)
        with torch.no_grad():
            train_temp_latent = temp_autoencoder_model.encoder(train_temps_tensor).cpu().numpy()  # (num_train_days*5, latent_dim_temp)
        val_temps_tensor = torch.from_numpy(val_temps.reshape(-1, window_length)).float().to(device)
        with torch.no_grad():
            val_temp_latent = temp_autoencoder_model.encoder(val_temps_tensor).cpu().numpy()
        test_temps_tensor = torch.from_numpy(test_temps.reshape(-1, window_length)).float().to(device)
        with torch.no_grad():
            test_temp_latent = temp_autoencoder_model.encoder(test_temps_tensor).cpu().numpy()
        
        train_ghis_tensor = torch.from_numpy(train_ghis.reshape(-1, window_length)).float().to(device)
        with torch.no_grad():
            train_ghi_latent = ghi_autoencoder_model.encoder(train_ghis_tensor).cpu().numpy()
        val_ghis_tensor = torch.from_numpy(val_ghis.reshape(-1, window_length)).float().to(device)
        with torch.no_grad():
            val_ghi_latent = ghi_autoencoder_model.encoder(val_ghis_tensor).cpu().numpy()
        test_ghis_tensor = torch.from_numpy(test_ghis.reshape(-1, window_length)).float().to(device)
        with torch.no_grad():
            test_ghi_latent = ghi_autoencoder_model.encoder(test_ghis_tensor).cpu().numpy()
        
        # Average latent features across 5 sites for temperature and GHI
        latent_dim_temp = train_temp_latent.shape[1]
        latent_dim_ghi = train_ghi_latent.shape[1]
        train_temp_avg = train_temp_latent.reshape(-1, 5, latent_dim_temp).mean(axis=1)
        val_temp_avg = val_temp_latent.reshape(-1, 5, latent_dim_temp).mean(axis=1)
        test_temp_avg = test_temp_latent.reshape(-1, 5, latent_dim_temp).mean(axis=1)
        
        train_ghi_avg = train_ghi_latent.reshape(-1, 5, latent_dim_ghi).mean(axis=1)
        val_ghi_avg = val_ghi_latent.reshape(-1, 5, latent_dim_ghi).mean(axis=1)
        test_ghi_avg = test_ghi_latent.reshape(-1, 5, latent_dim_ghi).mean(axis=1)
        
        # Expand these averages to hourly level
        train_temp_expanded = np.repeat(train_temp_avg, window_length, axis=0)
        val_temp_expanded = np.repeat(val_temp_avg, window_length, axis=0)
        test_temp_expanded = np.repeat(test_temp_avg, window_length, axis=0)
        
        train_ghi_expanded = np.repeat(train_ghi_avg, window_length, axis=0)
        val_ghi_expanded = np.repeat(val_ghi_avg, window_length, axis=0)
        test_ghi_expanded = np.repeat(test_ghi_avg, window_length, axis=0)
        
        # Combine latent features: For each hour, concatenate load, temperature, and GHI latent vectors.
        train_combined_latent = np.concatenate([train_latent_expanded, train_temp_expanded, train_ghi_expanded], axis=1)
        val_combined_latent = np.concatenate([val_latent_expanded, val_temp_expanded, val_ghi_expanded], axis=1)
        test_combined_latent = np.concatenate([test_latent_expanded, test_temp_expanded, test_ghi_expanded], axis=1)
    else:
        # Not using advanced mode: use only load latent
        train_combined_latent = train_latent_expanded
        val_combined_latent = val_latent_expanded
        test_combined_latent = test_latent_expanded
    
    # Append seasonal features (sin, cos for day of year)
    def month_day_to_day_of_year(month_day_array):
        days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        cum_days = np.cumsum(days_in_month)
        result = np.zeros(len(month_day_array))
        for i, (month, day) in enumerate(month_day_array):
            result[i] = cum_days[int(month)-1] + day
        return result / 365.0 * 2 * np.pi
    
    train_month_day = actual_train_df[['Month', 'Day']].values
    val_month_day = val_df[['Month', 'Day']].values
    test_month_day = test_df[['Month', 'Day']].values
    
    train_day_of_year = month_day_to_day_of_year(train_month_day)
    val_day_of_year = month_day_to_day_of_year(val_month_day)
    test_day_of_year = month_day_to_day_of_year(test_month_day)
    
    train_sin = np.sin(train_day_of_year).reshape(-1, 1)
    train_cos = np.cos(train_day_of_year).reshape(-1, 1)
    val_sin = np.sin(val_day_of_year).reshape(-1, 1)
    val_cos = np.cos(val_day_of_year).reshape(-1, 1)
    test_sin = np.sin(test_day_of_year).reshape(-1, 1)
    test_cos = np.cos(test_day_of_year).reshape(-1, 1)
    
    # Append seasonal features to static features
    train_static = np.concatenate([train_static, train_sin, train_cos], axis=1)
    val_static = np.concatenate([val_static, val_sin, val_cos], axis=1)
    test_static = np.concatenate([test_static, test_sin, test_cos], axis=1)
    
    # Finally, create final feature arrays by concatenating static and combined latent features
    train_final = np.concatenate([train_static, train_combined_latent], axis=1)
    val_final = np.concatenate([val_static, val_combined_latent], axis=1)
    test_final = np.concatenate([test_static, test_combined_latent], axis=1)
    
    print(f"Train static shape: {train_static.shape}, Train combined latent shape: {train_combined_latent.shape}")
    print(f"Val static shape: {val_static.shape}, Val combined latent shape: {val_combined_latent.shape}")
    print(f"Test static shape: {test_static.shape}, Test combined latent shape: {test_combined_latent.shape}")
    
    # Save processed arrays
    np.save(os.path.join(folder, 'train_final.npy'), train_final)
    np.save(os.path.join(folder, 'val_final.npy'), val_final)
    np.save(os.path.join(folder, 'test_final.npy'), test_final)
    np.save(os.path.join(folder, 'train_targets.npy'), train_targets)
    np.save(os.path.join(folder, 'val_targets.npy'), val_targets)
    np.save(os.path.join(folder, 'test_targets.npy'), test_targets)
    
    # Also save raw hourly loads for Year 1 and Year 2
    year1_hourly = train_df[train_df['Year'] == 1]['Load'].values.reshape(-1)
    year2_hourly = train_df[train_df['Year'] == 2]['Load'].values.reshape(-1)
    np.save(os.path.join(folder, 'year1_hourly.npy'), year1_hourly)
    np.save(os.path.join(folder, 'year2_hourly.npy'), year2_hourly)
    
    # Save daily averages (grouped by Month, Day)
    year1_daily = train_df[train_df['Year'] == 1].groupby(['Month', 'Day'])['Load'].mean().values
    year2_daily = train_df[train_df['Year'] == 2].groupby(['Month', 'Day'])['Load'].mean().values
    np.save(os.path.join(folder, 'year1_daily.npy'), year1_daily)
    np.save(os.path.join(folder, 'year2_daily.npy'), year2_daily)
    
    print("Preprocessing completed with cross-validation split.")
    return folder

def load_load_data_daywise(dataset, batch_size=64, shuffle=True):
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception("Processed data not found.")
    # Load daywise load data (assuming it was saved as (num_days, window_length))
    train_loads = np.load(os.path.join(folder, 'train_loads.npy'))
    from torch.utils.data import TensorDataset, DataLoader
    train_tensor = torch.from_numpy(train_loads).float()
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, train_tensor

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--window_length', type=int, default=24, help='Time window length in hours')
    args = parser.parse_args()
    dataset = args.dataset
    from src.models import LoadAutoEncoder
    # Instantiate with new latent_dim and dynamic window_length
    autoencoder = LoadAutoEncoder(input_dim=args.window_length, latent_dim=16).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    autoencoder_path = os.path.join("checkpoints", f"LoadAutoEncoder_{dataset}.pt")
    print("Looking for autoencoder checkpoint at:", os.path.abspath(autoencoder_path))
    print("Checkpoint exists:", os.path.exists(autoencoder_path))
    if os.path.exists(autoencoder_path):
        autoencoder.load_state_dict(torch.load(autoencoder_path))
        autoencoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Loaded trained autoencoder.")
    else:
        print("Trained autoencoder not found. Please train it first.")
        sys.exit(1)
    preprocess_with_autoencoder(dataset, autoencoder, window_length=args.window_length)