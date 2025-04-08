# import os
# import sys
# import pandas as pd
# import numpy as np
# import pickle
# import json
# from src.folderconstants import *
# from shutil import copyfile
# import torch

# import matplotlib.pyplot as plt


# datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB', 'temp_ghi']
# wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

# def normalize3(a, min_a = None, max_a = None):
#     if min_a is None:
#         min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
#     return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


# def load_data(dataset):
#     folder = os.path.join(output_folder, dataset)
#     os.makedirs(folder, exist_ok=True)
#     if dataset == 'temp_ghi':
#         try:
#             print("Starting temp_ghi preprocessing for load autoencoder (raw load values)...")
#             dataset_folder = 'data/temp_ghi'
            
#             print(f"Reading file from {os.path.join(dataset_folder, 'training.xlsx')}")
#             df = pd.read_excel(os.path.join(dataset_folder, 'training.xlsx'))
#             print(f"DataFrame shape: {df.shape}")
            
#             target_col = ['Load']
            
#             print("Splitting into year 1 (training) and year 2 (testing)...")
#             year1_data = df[df['Year'] == 1]  # Training data
#             year2_data = df[df['Year'] == 2]  # Testing data
#             print("\nBefore processing:")
#             print("Year 1 first day year value:", year1_data.iloc[0]['Year'])
#             print("Year 2 first day year value:", year2_data.iloc[0]['Year'])
#             print(f"Year 1 data shape: {year1_data.shape}")
#             print(f"Year 2 data shape: {year2_data.shape}")
            
#             def reshape_load(data):
#                 # Extract load column and reshape into daily sequences (num_days, 24, 1)
#                 loads = data[target_col].values  # shape (n, 1)
#                 daily_loads = loads.reshape(-1, 24, 1)  # assume each day has 24 hours
#                 print(f"Reshaped loads shape: {daily_loads.shape}")
#                 return daily_loads
            
#             print("Processing training load data...")
#             train_loads = reshape_load(year1_data)
#             print("Processing testing load data...")
#             test_loads = reshape_load(year2_data)
            
#             print("Using raw load values (no normalization)...")
#             # Ensure the shape is (num_days, 24, 1)
#             train_loads = train_loads.reshape(-1, 24, 1)
#             test_loads = test_loads.reshape(-1, 24, 1)
            
#             print(f"Final shapes: train_loads {train_loads.shape}, test_loads {test_loads.shape}")
            
#             # (Optional) If no normalization is applied, we don't need to save min/max.
#             preprocessing_params = {}
            
#             print(f"Saving to folder: {folder}")
#             np.save(os.path.join(folder, 'train_loads.npy'), train_loads)
#             np.save(os.path.join(folder, 'test_loads.npy'), test_loads)
            
#             with open(os.path.join(folder, 'preprocessing_params.pkl'), 'wb') as f:
#                 pickle.dump(preprocessing_params, f)
            
#             print("Processing completed successfully!")
            
#         except Exception as e:
#             print(f"Error in temp_ghi load preprocessing: {str(e)}")
#             import traceback
#             print(traceback.format_exc())
#             raise e
        
# def preprocess_with_autoencoder(dataset, autoencoder_model):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#     folder = os.path.join(output_folder, dataset)
#     os.makedirs(folder, exist_ok=True)
    
#     dataset_folder = 'data/temp_ghi'
#     df = pd.read_excel(os.path.join(dataset_folder, 'training.xlsx'))
    
#     year1_data = df[df['Year'] == 1]  # Training data (year 1)
#     year2_data = df[df['Year'] == 2]  # Testing data (year 2)

#     # Define columns
#     temporal_cols = ['Month', 'Day', 'Hour']
#     temp_cols = ['Site-1 Temp', 'Site-2 Temp', 'Site-3 Temp', 'Site-4 Temp', 'Site-5 Temp']
#     ghi_cols = ['Site-1 GHI', 'Site-2 GHI', 'Site-3 GHI', 'Site-4 GHI', 'Site-5 GHI']
#     target_col = ['Load']
    
#     feature_cols = temporal_cols + temp_cols + ghi_cols
    
#     # Static features at hourly resolution
#     train_features = year1_data[feature_cols].values  # shape: (num_days*24, 13)
#     test_features = year2_data[feature_cols].values   # shape: (num_days*24, 13)
    
#     # Load values reshaped to daily sequences (for autoencoder input)
#     train_loads = year1_data[target_col].values.reshape(-1, 24)  # (num_days, 24)
#     test_loads = year2_data[target_col].values.reshape(-1, 24)   # (num_days, 24)
    
#     # Calculate daily average load (target)
#     train_targets = train_loads.mean(axis=1, keepdims=True)  # shape: (num_days, 1)
#     test_targets = test_loads.mean(axis=1, keepdims=True)    # shape: (num_days, 1)
    
#     np.save(os.path.join(folder, 'train_targets.npy'), train_targets)
#     np.save(os.path.join(folder, 'test_targets.npy'), test_targets)
    
#     # Convert load data to torch tensors and pass through the encoder

#     trainX = torch.from_numpy(train_loads).float().to(device)
#     testX = torch.from_numpy(test_loads).float().to(device)
    
#     autoencoder_model.eval()
#     with torch.no_grad():
#         train_latent = autoencoder_model.encoder(trainX).cpu().numpy()  # (num_days, 8)
#         test_latent = autoencoder_model.encoder(testX).cpu().numpy()    # (num_days, 8)
    
#     # Expand latent features to hourly level by repeating each day's latent vector 24 times.
#     train_latent_expanded = np.repeat(train_latent, 24, axis=0)  # (num_days*24, 8)
#     test_latent_expanded = np.repeat(test_latent, 24, axis=0)    # (num_days*24, 8)
    
#     # Reshape static features (they are already hourly)
#     train_static = train_features.reshape(-1, 13)  # (num_days*24, 13)
#     test_static = test_features.reshape(-1, 13)    # (num_days*24, 13)
    
#     # Concatenate static and latent features: result has 13 + 8 = 21 columns per hourly sample.
#     train_final = np.concatenate([train_static, train_latent_expanded], axis=1)  # (8784, 21)
#     test_final = np.concatenate([test_static, test_latent_expanded], axis=1)     # (8760, 21)
    
#     # Save combined features
#     np.save(os.path.join(folder, 'train_final.npy'), train_final)
#     np.save(os.path.join(folder, 'test_final.npy'), test_final)
    
#     print("Shape of static features:", train_static.shape)  # (num_days*24, 13)
#     print("Shape of latent features:", train_latent.shape)    # (num_days, 8)
#     print("Expected combined shape:", train_static.shape[0], 21)
#     print("Actual combined shape:", train_final.shape)
#     print(f"Shape of train_final: {train_final.shape}")
#     print(f"Shape of test_final: {test_final.shape}")
#     print("Preprocessing completed: Combined static + latent features and targets saved.")

# if __name__ == '__main__':
#     commands = sys.argv[1:]
#     if len(commands) > 0:
#         for d in commands:
#             load_data(d)
#     else:
#         print("Usage: python preprocess.py <datasets>")
#         print(f"where <datasets> is space separated list of {datasets}")

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import torch
from src.folderconstants import output_folder
import matplotlib.pyplot as plt

# You may keep normalize3 here for other features—but we won't use it on load.
def normalize3(a, min_a=None, max_a=None):
    if min_a is None:
        min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def load_training_data(dataset):
    dataset_folder = os.path.join("data", dataset)
    df = pd.read_excel(os.path.join(dataset_folder, "training.xlsx"))
    print(f"Reading training data from: {os.path.join(dataset_folder, 'training.xlsx')}")
    print(f"DataFrame shape: {df.shape}")
    #train_df = df[df['Year'].isin([1, 2])]            #uncomment for year 3 regression predictioj
    train_df = df[df['Year'] == 1]                     #comment this and the next line 
    test_df = df[df['Year'] == 2]                     #comment out for regreeion
    print(f"Using training data for Year 1 and 2: {train_df.shape}")
    return train_df

def load_testing_data(dataset):
    dataset_folder = os.path.join("data", dataset)
    df = pd.read_excel(os.path.join(dataset_folder, "testing.xlsx"))
    print(f"Reading testing data from: {os.path.join(dataset_folder, 'testing.xlsx')}")
    print(f"DataFrame shape: {df.shape}")
    return df

def preprocess_with_autoencoder(dataset, autoencoder_model, temp_autoencoder_model=None, ghi_autoencoder_model=None):
    """
    Preprocesses data for the hourly autoencoder+regression pipeline with cross-validation.
    Uses existing Year/Month/Day columns to maintain day structure.
    
    Parameters:
    - dataset: Name of the dataset
    - autoencoder_model: Main autoencoder for load data
    - temp_autoencoder_model: Optional autoencoder for temperature data
    - ghi_autoencoder_model: Optional autoencoder for GHI data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    
    # Flag to check if we're using advanced features (temp and GHI encoding)
    use_advanced = (temp_autoencoder_model is not None and ghi_autoencoder_model is not None)
    if use_advanced:
        print("Using advanced feature extraction with temperature and GHI autoencoders")
    
    # Load dataframes
    train_df = load_training_data(dataset)
    test_df = load_testing_data(dataset)
    
    # Define columns
    temporal_cols = ['Month', 'Day', 'Hour']
    temp_cols = ['Site-1 Temp', 'Site-2 Temp', 'Site-3 Temp', 'Site-4 Temp', 'Site-5 Temp']
    ghi_cols = ['Site-1 GHI', 'Site-2 GHI', 'Site-3 GHI', 'Site-4 GHI', 'Site-5 GHI']
    feature_cols = temporal_cols + temp_cols + ghi_cols  # 13 columns
    
    # Create a unique day identifier using Year, Month, and Day columns
    train_df['day_id'] = train_df['Year'] * 10000 + train_df['Month'] * 100 + train_df['Day']
    
    # Get unique day IDs
    unique_day_ids = train_df['day_id'].unique()
    
    # Shuffle the day IDs (not the hours)
    np.random.seed(42)
    shuffled_day_ids = np.random.permutation(unique_day_ids)
    
    # Split into 80% train, 20% validation by days
    split_idx = int(0.8 * len(shuffled_day_ids))
    train_day_ids = shuffled_day_ids[:split_idx]
    val_day_ids = shuffled_day_ids[split_idx:]
    
    # Split dataframes by day IDs
    actual_train_df = train_df[train_df['day_id'].isin(train_day_ids)]
    val_df = train_df[train_df['day_id'].isin(val_day_ids)]
    
    print(f"Split data: {len(actual_train_df)} training samples, {len(val_df)} validation samples")
    print(f"Training days: {len(train_day_ids)}, Validation days: {len(val_day_ids)}")
    
    # Process basic static features (for basic mode or to combine with encoded features)
    train_static = actual_train_df[feature_cols].values  # (num_hours_train, 13)
    val_static = val_df[feature_cols].values  # (num_hours_val, 13)
    test_static = test_df[feature_cols].values  # (num_hours_test, 13)
    
    # Extract hourly targets
    train_targets = actual_train_df['Load'].values  # hourly targets
    val_targets = val_df['Load'].values  # hourly targets
    test_targets = np.zeros(test_df.shape[0])  # hourly targets (zeros)
    
    # Group training data by day for load encoding
    train_load_days = []
    train_temp_days = []
    train_ghi_days = []
    
    for day_id in train_day_ids:
        day_data = actual_train_df[actual_train_df['day_id'] == day_id].sort_values('Hour')
        
        if len(day_data) == 24:  # Only use complete days
            # Load data
            load_values = day_data['Load'].values
            train_load_days.append(load_values)
            
            # Temperature data (transpose to get shape (5, 24))
            if use_advanced:
                temp_values = day_data[temp_cols].values.T  # Shape: (5, 24)
                train_temp_days.append(temp_values)
                
                # GHI data (transpose to get shape (5, 24))
                ghi_values = day_data[ghi_cols].values.T  # Shape: (5, 24)
                train_ghi_days.append(ghi_values)
    
    train_loads = np.array(train_load_days)  # shape: (num_days_train, 24)
    
    if use_advanced:
        train_temps = np.vstack(train_temp_days)  # shape: (num_days_train*5, 24)
        train_ghis = np.vstack(train_ghi_days)  # shape: (num_days_train*5, 24)
    
    # Group validation data by day
    val_load_days = []
    val_temp_days = []
    val_ghi_days = []
    
    for day_id in val_day_ids:
        day_data = val_df[val_df['day_id'] == day_id].sort_values('Hour')
        
        if len(day_data) == 24:  # Only use complete days
            # Load data
            load_values = day_data['Load'].values
            val_load_days.append(load_values)
            
            # Temperature data
            if use_advanced:
                temp_values = day_data[temp_cols].values.T  # Shape: (5, 24)
                val_temp_days.append(temp_values)
                
                # GHI data
                ghi_values = day_data[ghi_cols].values.T  # Shape: (5, 24)
                val_ghi_days.append(ghi_values)
    
    val_loads = np.array(val_load_days)  # shape: (num_days_val, 24)
    
    if use_advanced:
        val_temps = np.vstack(val_temp_days)  # shape: (num_days_val*5, 24)
        val_ghis = np.vstack(val_ghi_days)  # shape: (num_days_val*5, 24)
    
    # Process test data
    test_days = len(test_df) // 24
    
    # Create improved test data using Year 1 and 2 data for same month/day if available
    test_load_days = []
    test_temp_days = []
    test_ghi_days = []
    
    # Create mapping of (month, day) to load patterns from Years 1 & 2
    month_day_to_load = {}
    month_day_to_temp = {}
    month_day_to_ghi = {}
    
    # Collect patterns from training data
    for day_id in train_day_ids:
        day_data = actual_train_df[actual_train_df['day_id'] == day_id].sort_values('Hour')
        if len(day_data) == 24:
            month = day_data['Month'].iloc[0]
            day = day_data['Day'].iloc[0]
            
            # Store load pattern
            load_pattern = day_data['Load'].values
            if (month, day) in month_day_to_load:
                month_day_to_load[(month, day)].append(load_pattern)
            else:
                month_day_to_load[(month, day)] = [load_pattern]
            
            if use_advanced:
                # Store temperature patterns
                temp_pattern = day_data[temp_cols].values.T
                if (month, day) in month_day_to_temp:
                    month_day_to_temp[(month, day)].append(temp_pattern)
                else:
                    month_day_to_temp[(month, day)] = [temp_pattern]
                
                # Store GHI patterns
                ghi_pattern = day_data[ghi_cols].values.T
                if (month, day) in month_day_to_ghi:
                    month_day_to_ghi[(month, day)].append(ghi_pattern)
                else:
                    month_day_to_ghi[(month, day)] = [ghi_pattern]
    
    # Add validation data too
    for day_id in val_day_ids:
        day_data = val_df[val_df['day_id'] == day_id].sort_values('Hour')
        if len(day_data) == 24:
            month = day_data['Month'].iloc[0]
            day = day_data['Day'].iloc[0]
            
            # Store load pattern
            load_pattern = day_data['Load'].values
            if (month, day) in month_day_to_load:
                month_day_to_load[(month, day)].append(load_pattern)
            else:
                month_day_to_load[(month, day)] = [load_pattern]
            
            if use_advanced:
                # Store temperature patterns
                temp_pattern = day_data[temp_cols].values.T
                if (month, day) in month_day_to_temp:
                    month_day_to_temp[(month, day)].append(temp_pattern)
                else:
                    month_day_to_temp[(month, day)] = [temp_pattern]
                
                # Store GHI patterns
                ghi_pattern = day_data[ghi_cols].values.T
                if (month, day) in month_day_to_ghi:
                    month_day_to_ghi[(month, day)].append(ghi_pattern)
                else:
                    month_day_to_ghi[(month, day)] = [ghi_pattern]
    
    # Calculate average patterns for each (month, day)
    avg_month_day_to_load = {}
    avg_month_day_to_temp = {}
    avg_month_day_to_ghi = {}
    
    for (month, day), patterns in month_day_to_load.items():
        avg_month_day_to_load[(month, day)] = np.mean(patterns, axis=0)
    
    if use_advanced:
        for (month, day), patterns in month_day_to_temp.items():
            avg_month_day_to_temp[(month, day)] = np.mean(patterns, axis=0)
        
        for (month, day), patterns in month_day_to_ghi.items():
            avg_month_day_to_ghi[(month, day)] = np.mean(patterns, axis=0)
    
    # Calculate global average patterns (fallback)
    global_avg_load = np.mean([p for patterns in month_day_to_load.values() for p in patterns], axis=0)
    
    if use_advanced:
        global_avg_temp = np.mean([p for patterns in month_day_to_temp.values() for p in patterns], axis=0)
        global_avg_ghi = np.mean([p for patterns in month_day_to_ghi.values() for p in patterns], axis=0)
    
    # Now create test loads/temps/ghis using historical data for same month/day
    for day_idx in range(test_days):
        start_idx = day_idx * 24
        end_idx = start_idx + 24
        day_data = test_df.iloc[start_idx:end_idx]
        
        if len(day_data) == 24:
            month = day_data['Month'].iloc[0]
            day = day_data['Day'].iloc[0]
            
            # Get load pattern for this month/day or fall back to global average
            if (month, day) in avg_month_day_to_load:
                test_load_days.append(avg_month_day_to_load[(month, day)])
            else:
                test_load_days.append(global_avg_load)
            
            if use_advanced:
                # Get temp pattern for this month/day
                if (month, day) in avg_month_day_to_temp:
                    test_temp_days.append(avg_month_day_to_temp[(month, day)])
                else:
                    test_temp_days.append(global_avg_temp)
                
                # Get GHI pattern for this month/day
                if (month, day) in avg_month_day_to_ghi:
                    test_ghi_days.append(avg_month_day_to_ghi[(month, day)])
                else:
                    test_ghi_days.append(global_avg_ghi)
    
    test_loads = np.array(test_load_days)  # shape: (num_days_test, 24)
    
    if use_advanced:
        test_temps = np.vstack(test_temp_days)  # shape: (num_days_test*5, 24)
        test_ghis = np.vstack(test_ghi_days)  # shape: (num_days_test*5, 24)
    
    print(f"Training loads shape: {train_loads.shape}")
    print(f"Validation loads shape: {val_loads.shape}")
    print(f"Test loads shape: {test_loads.shape}")
    
    if use_advanced:
        print(f"Training temps shape: {train_temps.shape}")
        print(f"Validation temps shape: {val_temps.shape}")
        print(f"Test temps shape: {test_temps.shape}")
        
        print(f"Training GHIs shape: {train_ghis.shape}")
        print(f"Validation GHIs shape: {val_ghis.shape}")
        print(f"Test GHIs shape: {test_ghis.shape}")
    
    # Extract latent features using autoencoders
    autoencoder_model.eval()
    
    with torch.no_grad():
        # Load data
        train_loads_tensor = torch.from_numpy(train_loads).float().to(device)
        train_load_latent = autoencoder_model.encoder(train_loads_tensor).cpu().numpy()
        
        val_loads_tensor = torch.from_numpy(val_loads).float().to(device)
        val_load_latent = autoencoder_model.encoder(val_loads_tensor).cpu().numpy()
        
        test_loads_tensor = torch.from_numpy(test_loads).float().to(device)
        test_load_latent = autoencoder_model.encoder(test_loads_tensor).cpu().numpy()
        
        if use_advanced:
            # Temperature data
            temp_autoencoder_model.eval()
            train_temps_tensor = torch.from_numpy(train_temps).float().to(device)
            train_temp_latent = temp_autoencoder_model.encoder(train_temps_tensor).cpu().numpy()
            
            val_temps_tensor = torch.from_numpy(val_temps).float().to(device)
            val_temp_latent = temp_autoencoder_model.encoder(val_temps_tensor).cpu().numpy()
            
            test_temps_tensor = torch.from_numpy(test_temps).float().to(device)
            test_temp_latent = temp_autoencoder_model.encoder(test_temps_tensor).cpu().numpy()
            
            # GHI data
            ghi_autoencoder_model.eval()
            train_ghis_tensor = torch.from_numpy(train_ghis).float().to(device)
            train_ghi_latent = ghi_autoencoder_model.encoder(train_ghis_tensor).cpu().numpy()
            
            val_ghis_tensor = torch.from_numpy(val_ghis).float().to(device)
            val_ghi_latent = ghi_autoencoder_model.encoder(val_ghis_tensor).cpu().numpy()
            
            test_ghis_tensor = torch.from_numpy(test_ghis).float().to(device)
            test_ghi_latent = ghi_autoencoder_model.encoder(test_ghis_tensor).cpu().numpy()
    
    # Create mappings from day_id to latent vectors
    train_day_id_to_load_latent = {}
    val_day_id_to_load_latent = {}
    
    # Map day_ids to load latent vectors
    for i, day_id in enumerate(train_day_ids):
        if i < len(train_load_latent):
            train_day_id_to_load_latent[day_id] = train_load_latent[i]
    
    for i, day_id in enumerate(val_day_ids):
        if i < len(val_load_latent):
            val_day_id_to_load_latent[day_id] = val_load_latent[i]
    
    if use_advanced:
        # For temp and GHI, we need to map each combination of day_id and site_id
        train_day_site_to_temp_latent = {}
        train_day_site_to_ghi_latent = {}
        val_day_site_to_temp_latent = {}
        val_day_site_to_ghi_latent = {}
        
        # Map to training data
        site_count = 5  # 5 sites
        for i, day_id in enumerate(train_day_ids):
            for site in range(site_count):
                latent_idx = i * site_count + site
                if latent_idx < len(train_temp_latent):
                    train_day_site_to_temp_latent[(day_id, site)] = train_temp_latent[latent_idx]
                    train_day_site_to_ghi_latent[(day_id, site)] = train_ghi_latent[latent_idx]
        
        # Map to validation data
        for i, day_id in enumerate(val_day_ids):
            for site in range(site_count):
                latent_idx = i * site_count + site
                if latent_idx < len(val_temp_latent):
                    val_day_site_to_temp_latent[(day_id, site)] = val_temp_latent[latent_idx]
                    val_day_site_to_ghi_latent[(day_id, site)] = val_ghi_latent[latent_idx]
    
    # Prepare latent features for each hour
    
    # Basic approach: use only load latent vectors
    if not use_advanced:
        # For each hour in train/val sets, find its day_id and use corresponding latent vector
        train_latent_expanded = []
        for day_id in actual_train_df['day_id']:
            if day_id in train_day_id_to_load_latent:
                train_latent_expanded.append(train_day_id_to_load_latent[day_id])
        train_latent_expanded = np.array(train_latent_expanded)
        
        val_latent_expanded = []
        for day_id in val_df['day_id']:
            if day_id in val_day_id_to_load_latent:
                val_latent_expanded.append(val_day_id_to_load_latent[day_id])
        val_latent_expanded = np.array(val_latent_expanded)
        
        # For test, expand each day's latent to 24 hours
        test_latent_expanded = np.repeat(test_load_latent, 24, axis=0)
    
    # Advanced approach: combine load, temp, and GHI latent vectors
    else:
        train_latent_expanded = []
        for _, row in actual_train_df.iterrows():
            day_id = row['day_id']
            hour = row['Hour']
            features = []
            
            # Add load latent if available
            if day_id in train_day_id_to_load_latent:
                features.append(train_day_id_to_load_latent[day_id])
            
            # Add temp and GHI latent for each site
            for site in range(site_count):
                if (day_id, site) in train_day_site_to_temp_latent:
                    features.append(train_day_site_to_temp_latent[(day_id, site)])
                if (day_id, site) in train_day_site_to_ghi_latent:
                    features.append(train_day_site_to_ghi_latent[(day_id, site)])
            
            # Concatenate all features if we have them
            if features:
                train_latent_expanded.append(np.concatenate(features))
        
        train_latent_expanded = np.array(train_latent_expanded)
        
        # Similar process for validation data
        val_latent_expanded = []
        for _, row in val_df.iterrows():
            day_id = row['day_id']
            hour = row['Hour']
            features = []
            
            # Add load latent if available
            if day_id in val_day_id_to_load_latent:
                features.append(val_day_id_to_load_latent[day_id])
            
            # Add temp and GHI latent for each site
            for site in range(site_count):
                if (day_id, site) in val_day_site_to_temp_latent:
                    features.append(val_day_site_to_temp_latent[(day_id, site)])
                if (day_id, site) in val_day_site_to_ghi_latent:
                    features.append(val_day_site_to_ghi_latent[(day_id, site)])
            
            # Concatenate all features if we have them
            if features:
                val_latent_expanded.append(np.concatenate(features))
        
        val_latent_expanded = np.array(val_latent_expanded)
        
        # For test data, construct combined features for each hour
        test_latent_expanded = []
        for day_idx in range(test_days):
            # Get latent vectors for this day
            load_latent = test_load_latent[day_idx]
            
            # Build a combined latent vector for each hour in this day
            for hour in range(24):
                features = [load_latent]
                
                # Add temp and GHI latent for each site
                for site in range(site_count):
                    temp_idx = day_idx * site_count + site
                    if temp_idx < len(test_temp_latent):
                        features.append(test_temp_latent[temp_idx])
                    
                    ghi_idx = day_idx * site_count + site
                    if ghi_idx < len(test_ghi_latent):
                        features.append(test_ghi_latent[ghi_idx])
                
                # Concatenate all features
                test_latent_expanded.append(np.concatenate(features))
        
        test_latent_expanded = np.array(test_latent_expanded)
    
    # Trim to ensure dimensions match
    min_len = min(len(train_static), len(train_latent_expanded))
    train_static = train_static[:min_len]
    train_latent_expanded = train_latent_expanded[:min_len]
    train_targets = train_targets[:min_len]
    
    min_len = min(len(val_static), len(val_latent_expanded))
    val_static = val_static[:min_len]
    val_latent_expanded = val_latent_expanded[:min_len]
    val_targets = val_targets[:min_len]
    
    min_len = min(len(test_static), len(test_latent_expanded))
    test_static = test_static[:min_len]
    test_latent_expanded = test_latent_expanded[:min_len]
    test_targets = test_targets[:min_len]
    
    # Print dimensions for debugging
    print(f"Train static shape: {train_static.shape}, Train latent expanded shape: {train_latent_expanded.shape}")
    print(f"Val static shape: {val_static.shape}, Val latent expanded shape: {val_latent_expanded.shape}")
    print(f"Test static shape: {test_static.shape}, Test latent expanded shape: {test_latent_expanded.shape}")
    
    # Add time-based seasonal features (sin/cos of day of year)
    train_month_day = actual_train_df[['Month', 'Day']].values
    val_month_day = val_df[['Month', 'Day']].values
    test_month_day = test_df[['Month', 'Day']].values
    
    # Convert month/day to day of year (approximation)
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cum_days = np.cumsum(days_in_month)
    
    def month_day_to_day_of_year(month_day_array):
        result = np.zeros(len(month_day_array))
        for i, (month, day) in enumerate(month_day_array):
            result[i] = cum_days[int(month)-1] + day
        return result / 365.0 * 2 * np.pi
    
    train_day_of_year = month_day_to_day_of_year(train_month_day)
    val_day_of_year = month_day_to_day_of_year(val_month_day)
    test_day_of_year = month_day_to_day_of_year(test_month_day)
    
    # Create sin/cos features
    train_sin = np.sin(train_day_of_year).reshape(-1, 1)
    train_cos = np.cos(train_day_of_year).reshape(-1, 1)
    val_sin = np.sin(val_day_of_year).reshape(-1, 1)
    val_cos = np.cos(val_day_of_year).reshape(-1, 1)
    test_sin = np.sin(test_day_of_year).reshape(-1, 1)
    test_cos = np.cos(test_day_of_year).reshape(-1, 1)
    
    # Add these features to static features
    train_static = np.concatenate([train_static, train_sin, train_cos], axis=1)
    val_static = np.concatenate([val_static, val_sin, val_cos], axis=1)
    test_static = np.concatenate([test_static, test_sin, test_cos], axis=1)
    
    # Concatenate static and latent features
    train_final = np.concatenate([train_static, train_latent_expanded], axis=1)
    val_final = np.concatenate([val_static, val_latent_expanded], axis=1)
    test_final = np.concatenate([test_static, test_latent_expanded], axis=1)
    
    # Save processed arrays
    np.save(os.path.join(folder, 'train_final.npy'), train_final)
    np.save(os.path.join(folder, 'val_final.npy'), val_final)
    np.save(os.path.join(folder, 'test_final.npy'), test_final)
    np.save(os.path.join(folder, 'train_targets.npy'), train_targets)
    np.save(os.path.join(folder, 'val_targets.npy'), val_targets)
    np.save(os.path.join(folder, 'test_targets.npy'), test_targets)
    
    # Also save raw hourly loads for Year 1 and Year 2
    year1_hourly = train_df[train_df['Year'] == 1]['Load'].values
    year2_hourly = train_df[train_df['Year'] == 2]['Load'].values
    np.save(os.path.join(folder, 'year1_hourly.npy'), year1_hourly)
    np.save(os.path.join(folder, 'year2_hourly.npy'), year2_hourly)
    
    # Save daily averages
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
    # Load the daywise load data (assume shape: (num_days, 24))
    train_loads = np.load(os.path.join(folder, 'train_loads.npy'))  # shape: (num_days, 24)
    # Create a tensor from the data
    train_tensor = torch.from_numpy(train_loads).float()
    # Create a dataset (no labels needed for autoencoder training)
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(train_tensor)
    # Create and return the DataLoader and also the tensor if needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, train_tensor


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    args = parser.parse_args()
    dataset = args.dataset
    from src.models import LoadAutoEncoder
    # Instantiate with latent_dim=16
    autoencoder = LoadAutoEncoder(input_dim=24, latent_dim=16).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    autoencoder_path = os.path.join("checkpoints", f"LoadAutoEncoder_{dataset}.pt")
    print("Looking for autoencoder checkpoint at:", os.path.abspath(autoencoder_path))
    print("Checkpoint exists:", os.path.exists(autoencoder_path))
    if os.path.exists(autoencoder_path):
        # Load checkpoint and then move to device
        autoencoder.load_state_dict(torch.load(autoencoder_path))
        autoencoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Loaded trained autoencoder.")
    else:
        print("Trained autoencoder not found. Please train it first.")
        sys.exit(1)
    preprocess_with_autoencoder(dataset, autoencoder)