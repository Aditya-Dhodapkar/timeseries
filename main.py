import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
from torch.utils.data import Dataset, DataLoader, TensorDataset
from preprocess import preprocess_with_autoencoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# from beepy import beep


import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='TranAD_Basic',
#                    help='Model to train')
# parser.add_argument('--dataset', type=str, default='temp_ghi',
#                    help='Dataset to use')
# parser.add_argument('--retrain', action='store_true',
#                    help='Whether to retrain the model')
# parser.add_argument('--test', action='store_true', help='Run in test mode')
# parser.add_argument('--advanced', action='store_true',
#                    help='Use advanced feature extraction with temp and GHI autoencoders')
# args = parser.parse_args()
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='LoadPredictor', help='Model to train: LoadAutoEncoder, TempAutoEncoder, GHIAutoEncoder, or LoadPredictor')
parser.add_argument('--dataset', type=str, default='temp_ghi', help='Dataset to use')
parser.add_argument('--retrain', action='store_true', help='Whether to retrain the model')
parser.add_argument('--advanced', action='store_true', help='Use advanced feature extraction with temp and GHI autoencoders')
parser.add_argument('--use_patchtst', action='store_true', help='Train and evaluate with PatchTST model')
parser.add_argument('--window_length', type=int, default=24, help='Total time‑steps in your input window (e.g. 48, 96, 168)')
parser.add_argument('--patch_size', type=int, default=24, help='Number of time steps per patch (for PatchTST)')
parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer encoder layers in PatchTST')
parser.add_argument('--d_model', type=int, default=64, help='Embedding dimension for PatchTST')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in PatchTST')
parser.add_argument('--epochs_patchtst', type=int, default=30, help='Epochs to train PatchTST')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    
    # Reshape data to 2D if it's 3D
    if len(data.shape) == 3:
        # Reshape to (days*hours, features)
        data = data.reshape(-1, data.shape[-1])
    
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i-w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
    
    return torch.stack(windows)

# def load_dataset(dataset): for TEMP & GHI ------------------------------------->>>>>>>>>>>>>>>>>>>>>
#     if dataset == 'temp_ghi':
#         folder = os.path.join(output_folder, dataset)
#         if not os.path.exists(folder):
#             raise Exception('Processed Data not found.')
        
#         # Load our preprocessed data
#         train_features = np.load(os.path.join(folder, 'train_features.npy'))
#         test_features = np.load(os.path.join(folder, 'test_features.npy'))
#         train_loads = np.load(os.path.join(folder, 'train_loads.npy'))
#         test_loads = np.load(os.path.join(folder, 'test_loads.npy'))
        
#         # Create dataloaders
#         train_loader = DataLoader(train_features, batch_size=train_features.shape[0])
#         test_loader = DataLoader(test_features, batch_size=test_features.shape[0])
        
#         # Dummy labels (needed for model compatibility)
#         labels = np.zeros_like(test_features[:,:,0]) # Only need 2D array for labels
        
#         return train_loader, test_loader, labels

def load_dataset(dataset):
    if dataset == 'temp_ghi':
        folder = os.path.join(output_folder, dataset)
        if not os.path.exists(folder):
            raise Exception('Processed Data not found.')
        
        # Load preprocessed data
        train_features = np.load(os.path.join(folder, 'train_features.npy'))  # (days, 24, 13)
        test_features = np.load(os.path.join(folder, 'test_features.npy'))
        train_loads = np.load(os.path.join(folder, 'train_loads.npy'))        # (days, 24, 1)
        test_loads = np.load(os.path.join(folder, 'test_loads.npy'))

        # Flatten from (days, 24, ...) to (days*24, ...)
        train_features = train_features.reshape(-1, train_features.shape[-1])  # shape -> (N, 13)
        test_features  = test_features.reshape(-1, test_features.shape[-1])
        train_loads    = train_loads.reshape(-1)  # shape -> (N,)
        test_loads     = test_loads.reshape(-1)

        # Convert to PyTorch tensors
        trainX = torch.from_numpy(train_features).float()
        trainY = torch.from_numpy(train_loads).float()    # shape (N,)
        testX  = torch.from_numpy(test_features).float()
        testY  = torch.from_numpy(test_loads).float()

        # Create TensorDatasets
        train_dataset = TensorDataset(trainX, trainY)
        test_dataset  = TensorDataset(testX,  testY)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

        # No need for 'labels' in regression, so just return None or an empty array
        return train_loader, test_loader, None
    
#loading data for load data
def load_load_data(dataset):
    # Assumes the preprocessing step has been run and saved the .npy files.
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    
    # Reload the original Excel file to re-split by year
    dataset_folder = os.path.join("data", dataset)
    df = pd.read_excel(os.path.join(dataset_folder, "training.xlsx"))
    
    # Use Year 1 for training, Year 2 for testing.
    train_data = df[df['Year'] == 1]
    test_data = df[df['Year'] == 2]
    
    target_col = ['Load']
    
    def reshape_load(data):
        loads = data[target_col].values  # shape (n, 1)
        daily_loads = loads.reshape(-1, 24, 1)  # assume each day has 24 hours
        return daily_loads
    
    train_loads = reshape_load(train_data)  # shape: (num_days_train, 24, 1)
    test_loads = reshape_load(test_data)    # shape: (num_days_test, 24, 1)
    
    # Squeeze out the last dimension so each sample is a 24-dimensional vector
    train_loads = train_loads.squeeze(-1)  # shape: (num_days_train, 24)
    test_loads = test_loads.squeeze(-1)    # shape: (num_days_test, 24)
    
    # Convert to PyTorch tensors
    trainX = torch.from_numpy(train_loads).float()
    testX = torch.from_numpy(test_loads).float()
    
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(trainX)
    test_dataset = TensorDataset(testX)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader


def load_load_data_daywise(dataset):
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')

    # load preprocessed day-level load data
    train_loads = np.load(os.path.join(folder, 'train_loads.npy'))  # shape (365, 24)
    test_loads  = np.load(os.path.join(folder, 'test_loads.npy'))   # shape (some_days, 24)

    # We want to train on Year 1, test on Year 2
    # so train_loads is shape (365, 24) -> each day is a sample
    trainX = torch.from_numpy(train_loads).float()  # shape (365, 24)
    testX  = torch.from_numpy(test_loads).float()   # shape (days_in_year2, 24)

    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(trainX)  # no labels needed, autoencoder
    test_dataset  = TensorDataset(testX)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, test_loader

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	print(f"Saving model to {file_path}")
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)
	print("Model saved successfully")

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).to(device)  # Add .to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data_loader, optimizer, scheduler, training=True):
    # MSE loss for regression
    criterion = nn.MSELoss(reduction='mean')
    
    if training:
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch).squeeze(-1)  # shape (batch_size,)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(X_batch)
        
        scheduler.step()
        # Return average loss
        return running_loss / len(data_loader.dataset)
    else:
        model.eval()
        all_losses = []
        all_preds = []
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                preds = model(X_batch).squeeze(-1)
                loss = criterion(preds, y_batch)
                all_losses.append(loss.item() * len(X_batch))
                all_preds.append(preds.cpu().numpy())
        avg_loss = sum(all_losses) / len(data_loader.dataset)
        all_preds = np.concatenate(all_preds)
        return avg_loss, all_preds
		




# -----------------------------
# 1. Argument Parsing
# -----------------------------


# -----------------------------
# 2. Load or train autoencoders
# -----------------------------

if args.model in ['TempAutoEncoder', 'GHIAutoEncoder']:
    args.advanced = True
    print(f"Advanced mode automatically enabled for {args.model}")

# Load autoencoder model
autoencoder = LoadAutoEncoder(input_dim=24, latent_dim=16).to(device)
autoencoder_path = os.path.join("checkpoints", f"LoadAutoEncoder_{args.dataset}.pt")

# Initialize temperature and GHI autoencoders if using advanced mode
temp_autoencoder = None
ghi_autoencoder = None

if args.advanced:
    temp_autoencoder = TempAutoEncoder(input_dim=24, latent_dim=8).to(device)
    ghi_autoencoder = GHIAutoEncoder(input_dim=24, latent_dim=8).to(device)
    
    temp_autoencoder_path = os.path.join("checkpoints", f"TempAutoEncoder_{args.dataset}.pt")
    ghi_autoencoder_path = os.path.join("checkpoints", f"GHIAutoEncoder_{args.dataset}.pt")

# Load autoencoder model

autoencoder = LoadAutoEncoder(input_dim=24, latent_dim=16).to(device)
autoencoder_path = os.path.join("checkpoints", f"LoadAutoEncoder_{args.dataset}.pt")

# Initialize temperature and GHI autoencoders if using advanced mode
temp_autoencoder = None
ghi_autoencoder = None

if args.advanced:
    temp_autoencoder = TempAutoEncoder(input_dim=24, latent_dim=8).to(device)
    ghi_autoencoder = GHIAutoEncoder(input_dim=24, latent_dim=8).to(device)
    
    temp_autoencoder_path = os.path.join("checkpoints", f"TempAutoEncoder_{args.dataset}.pt")
    ghi_autoencoder_path = os.path.join("checkpoints", f"GHIAutoEncoder_{args.dataset}.pt")

# Handle model loading or training based on the specified model
if args.model == 'LoadAutoEncoder' or args.model == 'TempAutoEncoder' or args.model == 'GHIAutoEncoder':
    if args.model == 'LoadAutoEncoder':
        selected_model = autoencoder
        model_path = autoencoder_path
    elif args.model == 'TempAutoEncoder':
        if not args.advanced:
            print("Error: TempAutoEncoder requires --advanced flag")
            exit(1)
        selected_model = temp_autoencoder
        model_path = temp_autoencoder_path
    elif args.model == 'GHIAutoEncoder':
        if not args.advanced:
            print("Error: GHIAutoEncoder requires --advanced flag")
            exit(1)
        selected_model = ghi_autoencoder
        model_path = ghi_autoencoder_path
    
    if args.retrain:
        print(f"Retraining {args.model}. Skipping checkpoint loading.")
        # For now, just save the untrained model (you should add actual training here)
        torch.save(selected_model.state_dict(), model_path)
        print(f"Saved {args.model} to {model_path}")
    else:
        if os.path.exists(model_path):
            print(f"Loading {args.model} checkpoint from: {os.path.abspath(model_path)}")
            selected_model.load_state_dict(torch.load(model_path))
            selected_model.to(device)
            print(f"Loaded trained {args.model}.")
        else:
            print(f"Trained {args.model} not found. Please run with --retrain to train it first.")
            exit(1)

    print(f"{args.model} operations completed.")
    exit(0)

# For LoadPredictor, load all required autoencoders
if args.retrain:
    print("Retraining autoencoders. Skipping checkpoint loading.")
    # Save the untrained models (actual training should be implemented)
    torch.save(autoencoder.state_dict(), autoencoder_path)
    print(f"Saved LoadAutoEncoder to {autoencoder_path}")
    
    if args.advanced:
        torch.save(temp_autoencoder.state_dict(), temp_autoencoder_path)
        torch.save(ghi_autoencoder.state_dict(), ghi_autoencoder_path)
        print(f"Saved TempAutoEncoder to {temp_autoencoder_path}")
        print(f"Saved GHIAutoEncoder to {ghi_autoencoder_path}")
else:
    # Load main autoencoder
    if os.path.exists(autoencoder_path):
        print("Looking for autoencoder checkpoint at:", os.path.abspath(autoencoder_path))
        autoencoder.load_state_dict(torch.load(autoencoder_path))
        autoencoder.to(device)
        print("Loaded trained LoadAutoEncoder.")
    else:
        print("Trained LoadAutoEncoder not found. Please run with --retrain to train it first.")
        exit(1)
    
    # Load temperature and GHI autoencoders if using advanced mode
    if args.advanced:
        if os.path.exists(temp_autoencoder_path) and os.path.exists(ghi_autoencoder_path):
            temp_autoencoder.load_state_dict(torch.load(temp_autoencoder_path))
            ghi_autoencoder.load_state_dict(torch.load(ghi_autoencoder_path))
            temp_autoencoder.to(device)
            ghi_autoencoder.to(device)
            print("Loaded trained TempAutoEncoder and GHIAutoEncoder.")
        else:
            print("Trained TempAutoEncoder or GHIAutoEncoder not found. Please run with --retrain to train them first.")
            exit(1)

# Process data with cross-validation
folder = preprocess_with_autoencoder(args.dataset, autoencoder, temp_autoencoder, ghi_autoencoder)

# -----------------------------
# 3. Load preprocessed data
# -----------------------------
train_features = np.load(os.path.join(folder, 'train_final.npy'))
val_features = np.load(os.path.join(folder, 'val_final.npy'))
test_features = np.load(os.path.join(folder, 'test_final.npy'))
train_targets = np.load(os.path.join(folder, 'train_targets.npy'))
val_targets = np.load(os.path.join(folder, 'val_targets.npy'))
test_targets = np.load(os.path.join(folder, 'test_targets.npy'))

print(f"Train features shape: {train_features.shape}")
print(f"Validation features shape: {val_features.shape}")
print(f"Test features shape: {test_features.shape}")

# -----------------------------
# 4. Normalize the features and targets
# -----------------------------
# Normalize features
features_min, features_max = np.min(train_features, axis=0), np.max(train_features, axis=0)
train_features_norm = (train_features - features_min) / (features_max - features_min + 1e-6)
val_features_norm = (val_features - features_min) / (features_max - features_min + 1e-6)
test_features_norm = (test_features - features_min) / (features_max - features_min + 1e-6)

# Normalize targets (only for training)
targets_min, targets_max = np.min(train_targets), np.max(train_targets)
train_targets_norm = (train_targets - targets_min) / (targets_max - targets_min + 1e-6)
val_targets_norm = (val_targets - targets_min) / (targets_max - targets_min + 1e-6)

print(f"Feature normalization range: {features_min[:5]} to {features_max[:5]} (showing first 5)")
print(f"Target normalization range: {targets_min} to {targets_max}")

# Save normalization parameters
normalization_params = {
    'features_min': features_min,
    'features_max': features_max,
    'targets_min': targets_min,
    'targets_max': targets_max
}
with open(os.path.join(folder, 'normalization_params.pkl'), 'wb') as f:
    pickle.dump(normalization_params, f)

# -----------------------------
# 5. Prepare data loaders
# -----------------------------
trainX = torch.from_numpy(train_features_norm).float()
trainY = torch.from_numpy(train_targets_norm).float().unsqueeze(1)
valX = torch.from_numpy(val_features_norm).float()
valY = torch.from_numpy(val_targets_norm).float().unsqueeze(1)
testX = torch.from_numpy(test_features_norm).float()

train_dataset = TensorDataset(trainX, trainY)
val_dataset = TensorDataset(valX, valY)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

## -----------------------------
# 6. Train models
# -----------------------------
# Enhanced training process
if args.model == 'PatchTST':
    # Better hyperparameters
    window_length = 72      # Use 3 days of data to predict next day
    patch_size = 6          # Smaller patches (e.g., 6-hour patches)
    num_layers = 4          # More transformer layers
    d_model = 256           # Larger embedding dimension
    num_heads = 8           # More attention heads
    batch_size = 16         # Smaller batch size for better gradient estimates
    epochs_patchtst = 100   # Train for more epochs with early stopping
    
    # Create enhanced train/test split with normalization
    # Extract features and targets for each day
    # Structure: use past 3 days to predict next day
    X_data = []
    y_data = []
    
    # Calculate max samples to create
    valid_samples = train_features_norm.shape[0] - window_length - 24
    
    # Create training samples with sliding window
    for i in range(0, valid_samples, 6):  # Step by 6 hours to avoid too much overlap
        X_data.append(train_features_norm[i:i+window_length])
        y_data.append(train_targets_norm[i+window_length:i+window_length+24])
    
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"Training data shape: X={X_data.shape}, y={y_data.shape}")
    
    # Create proper train/validation split
    indices = np.arange(len(X_data))
    np.random.shuffle(indices)
    split = int(0.85 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    X_train, y_train = X_data[train_indices], y_data[train_indices]
    X_val, y_val = X_data[val_indices], y_data[val_indices]
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create enhanced model
    model = PatchTSTModel(
        input_dim=train_features_norm.shape[1],
        window_length=window_length,
        patch_size=patch_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.2
    ).to(device)
    
    # Use a more sophisticated optimizer and learning rate schedule
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Learning rate scheduler with warmup
    def get_lr_scheduler(optimizer, warmup_steps, max_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(max_steps - step) / float(max(1, max_steps - warmup_steps)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    total_steps = epochs_patchtst * len(train_loader)
    warmup_steps = total_steps // 10
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    epoch_losses = []
    
    # Training loop
    for ep in range(epochs_patchtst):
        # Training
        model.train()
        train_loss = 0.0
        for step, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            outputs = model(bx)
            
            # Use Huber loss for robustness to outliers
            loss = F.huber_loss(outputs, by, delta=1.0)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * bx.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                outputs = model(bx)
                loss = F.mse_loss(outputs, by)  # Use MSE for validation to get true error
                val_loss += loss.item() * bx.size(0)
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        epoch_losses.append((avg_train_loss, avg_val_loss))
        
        print(f"PatchTST Epoch {ep+1}/{epochs_patchtst}  Train Loss {avg_train_loss:.6f}  Val Loss {avg_val_loss:.6f}")
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"checkpoints/PatchTST_{args.dataset}_best.pt")
            patience_counter = 0
            print(f"  New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(f"checkpoints/PatchTST_{args.dataset}_best.pt"))
    
    # Testing on Year 2 data
    test_start_idx = 0  # Start from the beginning of Year 2 data
    y2_features = test_features_norm[test_start_idx:test_start_idx+window_length]
    y2_input = torch.from_numpy(y2_features).float().unsqueeze(0).to(device)
    
    # Get Year 2 actual values for comparison
    dataset_folder = os.path.join("data", args.dataset)
    df = pd.read_excel(os.path.join(dataset_folder, "training.xlsx"))
    year2_data = df[df['Year'] == 2]
    year2_hourly = year2_data['Load'].values
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        y2_pred_norm = model(y2_input).squeeze(0).cpu().numpy()
    
    # Denormalize predictions
    y2_predictions = y2_pred_norm * (targets_max - targets_min) + targets_min
    
    # Get the actual values for the next 24 hours
    actual_start = window_length  # Start after the window used for input
    y2_actuals = year2_hourly[actual_start:actual_start+24]
    
    # Print comparison table
    print("\nHours 48 - 72: Year 2 Comparison (Actual vs Predicted):")
    header = f"{'Hour':>4} {'Actual':>8} {'Predicted':>10} {'Difference':>12} {'% Error':>10}"
    print(header)
    print("-" * len(header))

    for hour in range(24):
        actual    = y2_actuals[hour]
        predicted = y2_predictions[hour]
        diff      = predicted - actual
        pct_error = (diff / actual * 100) if actual != 0 else float('nan')
        print(f"{hour:<4} "
              f"{actual:<8.1f} "
              f"{predicted:<10.1f} "
              f"{diff:<12.1f} "
              f"{pct_error:<9.2f}%")
    # Calculate metrics
    mae = mean_absolute_error(y2_actuals, y2_predictions)
    rmse = np.sqrt(mean_squared_error(y2_actuals, y2_predictions))
    r2 = r2_score(y2_actuals, y2_predictions)
    
    print(f"\nDay 3 -  24 Hours Summary Statistics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Plot predictions vs actuals
    plt.figure(figsize=(12, 6))
    plt.plot(y2_actuals, label='Actual', marker='o', alpha=0.7)
    plt.plot(y2_predictions, label='Predicted', marker='x', alpha=0.7)
    plt.title('PatchTST: Actual vs Predicted Load (First 24 Hours)')
    plt.xlabel('Hour')
    plt.ylabel('Load')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join("plots", "patchtst_predictions.png"))
    plt.close()
    
    print("PatchTST model training and evaluation completed successfully.")
    sys.exit(0)
    
else:
    # Existing regression MLP branch
    print("\nTraining Load Predictor model...")
    model = LoadPredictor(input_dim=train_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    num_epochs = args.epochs
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batchX, batchY in train_loader:
            batchX, batchY = batchX.to(device), batchY.to(device)
            optimizer.zero_grad()
            predY = model(batchX)
            loss = criterion(predY, batchY)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batchX.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batchX, batchY in val_loader:
                batchX, batchY = batchX.to(device), batchY.to(device)
                predY = model(batchX)
                val_loss = criterion(predY, batchY)
                running_val_loss += val_loss.item() * batchX.size(0)
        
        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join("checkpoints", f"LoadPredictor_{args.dataset}_best.pt")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved at epoch {epoch+1}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Save autoencoders for future use
torch.save(autoencoder.state_dict(), autoencoder_path)
print(f"Saved LoadAutoEncoder to {autoencoder_path}")

if args.advanced:
    torch.save(temp_autoencoder.state_dict(), temp_autoencoder_path)
    torch.save(ghi_autoencoder.state_dict(), ghi_autoencoder_path)
    print(f"Saved TempAutoEncoder to {temp_autoencoder_path}")
    print(f"Saved GHIAutoEncoder to {ghi_autoencoder_path}")
# -----------------------------
# 7. Evaluate on validation data
# -----------------------------
print("\nEvaluating on validation data...")
# Load best model
if args.model == 'LoadPredictor':
    best_model_path = os.path.join("checkpoints", f"LoadPredictor_{args.dataset}_best.pt")
    model.load_state_dict(torch.load(best_model_path))
elif args.model == 'PatchTST':
    # This block shouldn't run due to the sys.exit() above, but just in case:
    best_model_path = os.path.join("checkpoints", f"PatchTST_{args.dataset}.pt")
    model.load_state_dict(torch.load(best_model_path))
model.eval()

val_predictions = []
val_actual = []
with torch.no_grad():
    for batchX, batchY in val_loader:
        batchX = batchX.to(device)
        
        # Get normalized predictions
        norm_predY = model(batchX)
        
        # Denormalize predictions
        raw_predY = norm_predY * (targets_max - targets_min) + targets_min
        
        val_predictions.append(raw_predY.cpu().numpy())
        val_actual.append(batchY.cpu().numpy() * (targets_max - targets_min) + targets_min)

# Concatenate all predictions and true values
val_predictions = np.concatenate(val_predictions).squeeze()
val_actual = np.concatenate(val_actual).squeeze()

# Compute metrics using raw values
mse = mean_squared_error(val_actual, val_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(val_actual, val_predictions)
r2 = r2_score(val_actual, val_predictions)

print(f"Validation Metrics on Raw Data:")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")  

# -----------------------------
# 8. Generate predictions for Year 3
# -----------------------------
print("\nGenerating predictions for Year 3...")
year3_predictions = []
with torch.no_grad():
    for i in range(0, len(testX), 64):
        batch = testX[i:i+64].to(device)
        norm_predY = model(batch)
        raw_predY = norm_predY * (targets_max - targets_min) + targets_min
        year3_predictions.append(raw_predY.cpu().numpy())

year3_predictions = np.concatenate(year3_predictions).squeeze()

# Save Year 3 predictions
np.save(os.path.join(folder, 'year3_predictions_hourly.npy'), year3_predictions)

# Reshape to daily values for plotting
num_days = year3_predictions.shape[0] // 24
year3_predictions_daily = year3_predictions.reshape(num_days, 24).mean(axis=1)
np.save(os.path.join(folder, 'year3_predictions_daily.npy'), year3_predictions_daily)

print(f"Year 3 predictions saved: {len(year3_predictions)} hourly values, {len(year3_predictions_daily)} daily averages")

# Calculate the average ratio between Year 1/2 and Year 3 predictions
year1_daily = np.load(os.path.join(folder, 'year1_daily.npy'))
year2_daily = np.load(os.path.join(folder, 'year2_daily.npy'))
year1_mean = year1_daily.mean()
year2_mean = year2_daily.mean()
year3_mean = year3_predictions_daily.mean()

# Scale factor (average of Year 1 & 2 divided by Year 3)
scale_factor = (year1_mean + year2_mean) / (2 * year3_mean)
print(f"Scaling Year 3 predictions by factor: {scale_factor:.2f}")

# Apply scaling
year3_predictions_scaled = year3_predictions * scale_factor
year3_predictions_daily_scaled = year3_predictions_daily * scale_factor

# Save scaled predictions
np.save(os.path.join(folder, 'year3_predictions_hourly_scaled.npy'), year3_predictions_scaled)
np.save(os.path.join(folder, 'year3_predictions_daily_scaled.npy'), year3_predictions_daily_scaled)


# -----------------------------


#FOR YEAR 2 PREDICITON


# Add after generating predictions
# Extract first 4 days (96 hours) of Year 2
dataset_folder = os.path.join("data", args.dataset)
df = pd.read_excel(os.path.join(dataset_folder, "training.xlsx"))
year2_hourly = df[df['Year'] == 2]['Load'].values
print(f"Year 2 hourly data shape (from Excel): {year2_hourly.shape}")
print(f"Year 2 hourly data exists: {len(year2_hourly) > 0}")

# Get first 96 hours (January 1-4)
first_4_days_actual = year2_hourly[:96]
first_4_days_predicted = year3_predictions[:96]  # Using the existing "year3_predictions" variable

# Print comparison table with fixed-width formatting for perfect alignment
print("\nFirst 4 Days - Year 2 Comparison (Actual vs Predicted):")
print(f"{'Hour':5} {'Day':5} {'Actual':10} {'Predicted':10} {'Difference':12} {'Percent Error':12}")
print("-" * 60)

for hour in range(96):
    day = hour // 24 + 1  # Day 1, 2, 3, or 4
    actual = first_4_days_actual[hour]
    predicted = first_4_days_predicted[hour]
    diff = predicted - actual
    pct_error = (diff / actual) * 100 if actual != 0 else float('inf')
    print(f"{hour:<5} {day:<5} {actual:<10.1f} {predicted:<10.1f} {diff:<12.1f} {pct_error:<10.2f}%")

# Calculate summary statistics for each day
for day in range(1, 5):
    start_hour = (day - 1) * 24
    end_hour = day * 24
    
    day_actual = first_4_days_actual[start_hour:end_hour]
    day_predicted = first_4_days_predicted[start_hour:end_hour]
    
    mae = mean_absolute_error(day_actual, day_predicted)
    rmse = np.sqrt(mean_squared_error(day_actual, day_predicted))
    r2 = r2_score(day_actual, day_predicted)
    
    print(f"\nDay {day} Summary Statistics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

# Overall statistics for all 4 days
mae = mean_absolute_error(first_4_days_actual, first_4_days_predicted)
rmse = np.sqrt(mean_squared_error(first_4_days_actual, first_4_days_predicted))
r2 = r2_score(first_4_days_actual, first_4_days_predicted)

print(f"\nOverall 4-Day Summary Statistics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")





#-----------------------------

# -----------------------------
# 9. Plot results
# -----------------------------
os.makedirs("plots", exist_ok=True)

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(os.path.join("plots", "loss_curves.png"))
plt.close()

# Plot validation predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(val_actual[:500], label='Actual Load', alpha=0.7)
plt.plot(val_predictions[:500], label='Predicted Load', alpha=0.7)
plt.xlabel('Hour')
plt.ylabel('Load')
plt.title('Validation Data: Actual vs Predicted Load (First 500 Hours)')
plt.legend()
plt.savefig(os.path.join("plots", "validation_predictions.png"))
plt.close()

# Plot validation scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(val_actual, val_predictions, alpha=0.3)
plt.plot([min(val_actual), max(val_actual)], [min(val_actual), max(val_actual)], 'r--')
plt.xlabel('Actual Load')
plt.ylabel('Predicted Load')
plt.title('Validation Data: Actual vs Predicted Load')
plt.savefig(os.path.join("plots", "validation_scatter.png"))
plt.close()

# Plot overall daily comparison (original)
plt.figure(figsize=(12, 6))
plt.plot(year1_daily, label="Year 1 Daily Load", alpha=0.7)
plt.plot(year2_daily, label="Year 2 Daily Load", alpha=0.7)
plt.plot(year3_predictions_daily, label="Year 3 Predicted Daily Load", alpha=0.7)
plt.xlabel("Day of Year")
plt.ylabel("Load")
plt.title("Overall Daily Load Comparison (Original)")
plt.legend()
plt.savefig(os.path.join("plots", "daily_comparison.png"))
plt.close()

# Plot overall daily comparison with scaled predictions
plt.figure(figsize=(12, 6))
plt.plot(year1_daily, label="Year 1 Daily Load", alpha=0.7)
plt.plot(year2_daily, label="Year 2 Daily Load", alpha=0.7)
plt.plot(year3_predictions_daily_scaled, label="Year 3 Predicted Daily Load (Scaled)", alpha=0.7)
plt.xlabel("Day of Year")
plt.ylabel("Load")
plt.title("Overall Daily Load Comparison (with Scaled Year 3)")
plt.legend()
plt.savefig(os.path.join("plots", "daily_comparison_scaled.png"))
plt.close()

# Create January 1-2 comparison with scaled predictions
# Extract January 1-2 (first 48 hours)
year1_hourly = np.load(os.path.join(folder, 'year1_hourly.npy'))
year2_hourly = np.load(os.path.join(folder, 'year2_hourly.npy'))
jan_1_2_year1 = year1_hourly[:48]
jan_1_2_year2 = year2_hourly[:48]
jan_1_2_year3 = year3_predictions_scaled[:48]

# Create x-axis (hours 0-47)
hours = np.arange(48)

# Plot January 1-2 comparison
plt.figure(figsize=(12, 6))
plt.plot(hours, jan_1_2_year1, label="Year 1 Actual", alpha=0.7)
plt.plot(hours, jan_1_2_year2, label="Year 2 Actual", alpha=0.7)
plt.plot(hours, jan_1_2_year3, label="Year 3 Predicted", alpha=0.7)
plt.xlabel("Hour (January 1-2)")
plt.ylabel("Load")
plt.title("January 1-2: Hourly Load Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("plots", "january_1_2_comparison.png"))
plt.close()

print("Done. To train PatchTST on Year1→Year2, run with:")
print("  python main.py --dataset temp_ghi --window_length 48 --use_patchtst")