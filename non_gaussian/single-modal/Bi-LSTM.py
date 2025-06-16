import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import copy
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pickle
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


class ContaminantDataset(Dataset):
    def __init__(self, time_series, params):
        self.time_series = time_series
        self.params = params

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, idx):
        return self.time_series[idx], self.params[idx]


def augment_for_location(X, Y, augment_factor=1.0, noise_level=0.01):
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y

    loc_params = Y_np[:, :2]

    loc_std = np.std(loc_params, axis=0)
    loc_mean = np.mean(loc_params, axis=0)

    important_indices = []
    for i in range(len(Y_np)):
        if (abs(loc_params[i, 0] - loc_mean[0]) > loc_std[0] or
                abs(loc_params[i, 1] - loc_mean[1]) > loc_std[1]):
            important_indices.append(i)

    # If no samples found, use random samples
    if not important_indices:
        important_indices = np.random.choice(
            len(Y_np), size=int(len(Y_np) * 0.2), replace=False)

    # Select samples to augment
    X_to_augment = X_np[important_indices]
    Y_to_augment = Y_np[important_indices]

    # Create augmented versions with noise
    num_augmentations = int(len(important_indices) * augment_factor)

    X_augmented_list = [X_np]
    Y_augmented_list = [Y_np]

    for _ in range(num_augmentations):
        # Random selection of samples to add noise to
        indices = np.random.choice(len(X_to_augment), size=len(X_to_augment), replace=True)

        # Create noisy versions
        X_noisy = X_to_augment[indices] + np.random.normal(0, noise_level, X_to_augment[indices].shape)
        Y_noisy = Y_to_augment[indices].copy()  # Keep targets the same

        X_augmented_list.append(X_noisy)
        Y_augmented_list.append(Y_noisy)

    # Combine original and augmented data
    X_combined = np.concatenate(X_augmented_list, axis=0)
    Y_combined = np.concatenate(Y_augmented_list, axis=0)

    # Convert back to tensors
    X_tensor = torch.tensor(X_combined, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_combined, dtype=torch.float32)

    return X_tensor, Y_tensor


def create_data_loaders(X_time_series, Y, batch_size=128, augment=True):
    # Apply data augmentation if enabled
    if augment:
        X_time_series, Y = augment_for_location(X_time_series, Y, augment_factor=0.3)

    full_dataset = ContaminantDataset(X_time_series, Y)

    def collate_fn(batch):
        btc_tensor = torch.stack([x[0] for x in batch]).unsqueeze(1).to(device)
        params_tensor = torch.stack([x[1] for x in batch]).to(device)
        return btc_tensor, params_tensor

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        'val': DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn),
        'test': DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    }
    return loaders['train'], loaders['val'], loaders['test']


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),  # Layer normalization
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Output feature dimension
        self.feature_dim = hidden_dim * 2

    def forward(self, x):
        # x shape: [batch, 1, seq_len]
        x = x.permute(0, 2, 1)  # to [batch, seq_len, 1]

        # Run LSTM
        outputs, _ = self.lstm(x)  # outputs: [batch, seq_len, hidden_dim*2]

        # Apply attention
        attn_weights = self.attention(outputs)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * outputs, dim=1)  # [batch, hidden_dim*2]

        return context

class SpecializedModel(nn.Module):
    """Model with specialized sub-networks for location and release parameters"""

    def __init__(self, input_dim=1, hidden_dim=128, dropout=0.3):
        super().__init__()
        # Shared feature extractor
        self.encoder = LSTMEncoder(input_dim, hidden_dim, dropout=dropout)
        encoder_output_dim = self.encoder.feature_dim  # hidden_dim * 2

        # Specialized network for location parameters (larger, more focused)
        self.location_net = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),  # Keep same dimension
            nn.LayerNorm(encoder_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 location parameters
        )

        # Network for release history parameters
        self.release_net = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),  # Less dropout for release parameters
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 release history parameters
        )

    def forward(self, x):
        # Extract shared features
        features = self.encoder(x)

        # Get specialized predictions
        loc_params = self.location_net(features)
        rel_params = self.release_net(features)

        # Combine predictions
        return torch.cat([loc_params, rel_params], dim=1)

    def get_location_predictions(self, x):
        """Get only location parameter predictions"""
        features = self.encoder(x)
        return self.location_net(features)

    def get_release_predictions(self, x):
        """Get only release parameter predictions"""
        features = self.encoder(x)
        return self.release_net(features)

def weighted_parameter_loss(pred, target):
    """
    Custom loss function that weights location parameters more heavily
    """
    # Higher weights for location parameters (1-2), lower for release parameters (3-6)
    param_weights = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0]).to(device)

    # Calculate squared errors
    squared_error = (pred - target) ** 2

    # Apply weights to each parameter
    weighted_squared_error = squared_error * param_weights.unsqueeze(0)

    # Return mean loss
    return torch.mean(weighted_squared_error)

def combined_loss_function(model, x, y):
    """
    Combined loss function with main loss and auxiliary loss for location parameters
    """
    # Get full predictions
    full_pred = model(x)

    # Main loss (weighted MSE)
    main_loss = weighted_parameter_loss(full_pred, y)

    # Get specialized location predictions and compute auxiliary loss
    loc_pred = full_pred[:, :2]  # First two parameters
    loc_true = y[:, :2]
    aux_loss = 2.0 * F.mse_loss(loc_pred, loc_true)  # Extra weight on location

    # Combine losses
    total_loss = main_loss + aux_loss

    return total_loss

def train_model(X_time_series, Y, num_epochs=500, patience=50, batch_size=128):
    # Initialize standardizers
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    # Standardize input data
    X_np = X_time_series.numpy()
    Y_np = Y.numpy()
    X_scaled = scaler_X.fit_transform(X_np)
    Y_scaled = scaler_Y.fit_transform(Y_np)

    # Convert to tensors
    X_time_series_scaled = torch.tensor(X_scaled, dtype=torch.float32)
    Y_scaled = torch.tensor(Y_scaled, dtype=torch.float32)

    # Create data loaders with augmentation
    train_loader, val_loader, test_loader = create_data_loaders(
        X_time_series_scaled, Y_scaled, batch_size=batch_size, augment=True)

    # Save standardizers
    with open("scaler_X_specialized.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open("scaler_Y_specialized.pkl", "wb") as f:
        pickle.dump(scaler_Y, f)
    print("Standard scalers saved as 'scaler_X_specialized.pkl' and 'scaler_Y_specialized.pkl'")

    # Initialize specialized model
    model = SpecializedModel().to(device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=10, verbose=True, min_lr=1e-6)

    best_loss, best_model = float('inf'), None
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for btc, params in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            # Use combined loss function with auxiliary loss
            loss = combined_loss_function(model, btc, params)

            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for btc, params in val_loader:
                val_loss += combined_loss_function(model, btc, params).item()
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save the best model
            torch.save({'model_state_dict': model.state_dict()}, 'best_specialized_model.pth')
            print(f"New best model saved with val_loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Load the best model
    model.load_state_dict(best_model)

    # Test evaluation
    model.eval()
    test_preds, test_true = [], []
    test_loc_preds, test_loc_true = [], []
    test_rel_preds, test_rel_true = [], []

    with torch.no_grad():
        for btc, params in test_loader:
            # Full predictions
            pred = model(btc)
            test_preds.append(pred.cpu().numpy())
            test_true.append(params.cpu().numpy())

            # Location-specific predictions
            loc_pred = pred[:, :2]
            loc_true = params[:, :2]
            test_loc_preds.append(loc_pred.cpu().numpy())
            test_loc_true.append(loc_true.cpu().numpy())

            # Release-specific predictions
            rel_pred = pred[:, 2:]
            rel_true = params[:, 2:]
            test_rel_preds.append(rel_pred.cpu().numpy())
            test_rel_true.append(rel_true.cpu().numpy())

    # Convert predictions and ground truth back to original scale
    test_preds_scaled = np.concatenate(test_preds)
    test_true_scaled = np.concatenate(test_true)
    test_preds = scaler_Y.inverse_transform(test_preds_scaled)
    test_true = scaler_Y.inverse_transform(test_true_scaled)

    # Process location and release predictions separately
    test_loc_preds_scaled = np.concatenate(test_loc_preds)
    test_loc_true_scaled = np.concatenate(test_loc_true)
    test_rel_preds_scaled = np.concatenate(test_rel_preds)
    test_rel_true_scaled = np.concatenate(test_rel_true)

    # Inverse transform for location and release parameters
    loc_columns = [0, 1]  # Assuming first 2 columns are location
    rel_columns = [2, 3, 4, 5]  # Assuming last 4 columns are release

    # Create empty arrays for the inverse transformed data
    test_loc_preds = np.zeros_like(test_loc_preds_scaled)
    test_loc_true = np.zeros_like(test_loc_true_scaled)
    test_rel_preds = np.zeros_like(test_rel_preds_scaled)
    test_rel_true = np.zeros_like(test_rel_true_scaled)

    # Apply inverse transform for each parameter individually
    for i, col in enumerate(loc_columns):
        mean = scaler_Y.mean_[col]
        std = scaler_Y.scale_[col]
        test_loc_preds[:, i] = test_loc_preds_scaled[:, i] * std + mean
        test_loc_true[:, i] = test_loc_true_scaled[:, i] * std + mean

    for i, col in enumerate(rel_columns):
        mean = scaler_Y.mean_[col]
        std = scaler_Y.scale_[col]
        test_rel_preds[:, i] = test_rel_preds_scaled[:, i] * std + mean
        test_rel_true[:, i] = test_rel_true_scaled[:, i] * std + mean

    # Calculate evaluation metrics for all parameters
    r2_scores = []
    mse_scores = []

    for i in range(test_preds.shape[1]):
        r2 = r2_score(test_true[:, i], test_preds[:, i])
        mse = mean_squared_error(test_true[:, i], test_preds[:, i])
        r2_scores.append(r2)
        mse_scores.append(mse)
        print(f"Parameter {i + 1}: R² = {r2:.4f}, MSE = {mse:.4f}")

    # Calculate metrics for location and release parameters separately
    loc_r2 = r2_score(test_loc_true, test_loc_preds)
    loc_mse = mean_squared_error(test_loc_true, test_loc_preds)
    rel_r2 = r2_score(test_rel_true, test_rel_preds)
    rel_mse = mean_squared_error(test_rel_true, test_rel_preds)

    print(f"\nLocation Parameters: R² = {loc_r2:.4f}, MSE = {loc_mse:.4f}")
    print(f"Release Parameters: R² = {rel_r2:.4f}, MSE = {rel_mse:.4f}")

    # Overall metrics
    test_metrics = {
        'r2': r2_score(test_true, test_preds),
        'mse': mean_squared_error(test_true, test_preds),
        'rmse': np.sqrt(mean_squared_error(test_true, test_preds)),
        'loc_r2': loc_r2,
        'loc_mse': loc_mse,
        'rel_r2': rel_r2,
        'rel_mse': rel_mse,
        'per_param_r2': r2_scores,
        'per_param_mse': mse_scores
    }

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig('training_history_specialized.png')

    # Plot predictions vs actual for each parameter
    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.scatter(test_true[:, i], test_preds[:, i], alpha=0.5)
        plt.plot([test_true[:, i].min(), test_true[:, i].max()],
                 [test_true[:, i].min(), test_true[:, i].max()], 'r--')
        plt.title(f'Parameter {i + 1}: R² = {r2_scores[i]:.4f}')
        plt.xlabel('True')
        plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig('predictions_vs_actual_enhanced.png')

    return model, history, test_metrics

if __name__ == "__main__":
    # Load data
    a = np.loadtxt('Data_treat/BTCs_InterpNew.txt')
    b = np.loadtxt('Data_treat/DataGCSI.txt')
    X_time_series = torch.tensor(a, dtype=torch.float32)
    Y = torch.tensor(b, dtype=torch.float32)

    # Train model
    model, history, metrics = train_model(X_time_series, Y, num_epochs=500, patience=50)
    print(f"\nOverall Test Metrics: R2={metrics['r2']:.4f}, MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}")

