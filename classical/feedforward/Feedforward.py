import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import io
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# Configure logging
log_dir = "/Users/ca5/Desktop/qnn_fnl/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"feedforward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FeedforwardNN")

# Create directories specific to the feedforward model
base_dir = os.path.dirname(os.path.abspath(__file__))
ff_graphs_dir = os.path.join(base_dir, "graphs")
ff_models_dir = os.path.join(base_dir, "models")
os.makedirs(ff_graphs_dir, exist_ok=True)
os.makedirs(ff_models_dir, exist_ok=True)
logger.info(f"Storing graphs in: {ff_graphs_dir}")
logger.info(f"Storing models in: {ff_models_dir}")

# Keep the original graphs directory for compatibility
graphs_dir = "/Users/ca5/Desktop/qnn_fnl/graphs" 
os.makedirs(graphs_dir, exist_ok=True)

# Set up device - use MPS if available (Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
logger.info(f"Using device: {device}")

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3):
        super(FeedforwardNN, self).__init__()
        
        # Build layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            # Add linear layer
            layers.append(nn.Linear(prev_dim, dim))
            
            # Add batch normalization (except for last layer)
            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(dim))
                
            # Add activation function
            layers.append(nn.LeakyReLU(0.2))
            
            # Add dropout (except for last layer)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = dim
        
        # Add output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def load_data(file_path, target_column="dGmix"):
    """Load and preprocess data, dropping non-numeric columns"""
    logger.info(f"Loading data from {file_path}")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Check if target column exists
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in dataset")
            logger.info(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)
            
        # Drop non-numeric columns
        original_shape = df.shape
        df = df.select_dtypes(include=['number'])
        logger.info(f"Dropped non-numeric columns. Shape before: {original_shape}, after: {df.shape}")
        
        # Ensure target column is still present
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' was dropped as it's not numeric")
            sys.exit(1)
            
        return df, target_column
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)

def preprocess_data(df, target_column):
    """Split and scale the data"""
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    feature_scaler = RobustScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale target
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    
    logger.info("Data preprocessing completed")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_train, y_test, feature_scaler, y_scaler

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, 
                patience=15, lr=0.001, weight_decay=1e-4):
    """Train the feedforward neural network"""
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    # Move model to device
    model = model.to(device)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss function and optimizer
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                    patience=5, verbose=True)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_model_path = os.path.join(ff_models_dir, "best_feedforward_model.pt")
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Mini-batch training with shuffle
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Explicit synchronization for Metal
            if device.type == "mps":
                torch.mps.synchronize()
                
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()
            val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            logger.info(f"Epoch {epoch+1}/{epochs}: New best model saved with val_loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Force garbage collection for memory management
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
            
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    logger.info(f"Loaded best model from {best_model_path}")
    
    return model, train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_test_orig, y_scaler):
    """Evaluate model performance"""
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    # Convert predictions back to original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_orig.values, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig.values, y_pred)
    r2 = r2_score(y_test_orig.values, y_pred)
    
    logger.info(f'Test MSE: {mse:.4f}')
    logger.info(f'Test RMSE: {rmse:.4f}')
    logger.info(f'Test MAE: {mae:.4f}')
    logger.info(f'Test R²: {r2:.4f}')
    
    # Get residuals for later plotting
    errors = y_pred.flatten() - y_test_orig.values
    
    return y_pred, mse, rmse, mae, r2, errors

def plot_actual_vs_predicted(y_test, y_pred, rmse):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test.values, y_pred.flatten(), alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted (RMSE: {rmse:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Add annotation
    plt.annotate(f'RMSE: {rmse:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    # Save to both directories
    plt.savefig(f'{ff_graphs_dir}/actual_vs_predicted_test.png', dpi=300)
    plt.savefig(f'{graphs_dir}/actual_vs_predicted_test.png', dpi=300)
    logger.info(f"Actual vs Predicted plot saved to {ff_graphs_dir}/actual_vs_predicted_test.png")
    plt.close()

def plot_qq_residuals(errors):
    """Plot Q-Q plot of residuals"""
    plt.figure(figsize=(10, 6))
    stats.probplot(errors, plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save to both directories
    plt.savefig(f'{ff_graphs_dir}/qq_plot_residuals.png', dpi=300)
    plt.savefig(f'{graphs_dir}/qq_plot_residuals.png', dpi=300)
    logger.info(f"Q-Q Plot of Residuals saved to {ff_graphs_dir}/qq_plot_residuals.png")
    plt.close()

def plot_confidence_intervals(model, X_test, y_test, y_scaler, n_bootstrap=100):
    """Create confidence intervals using bootstrap"""
    logger.info("Generating bootstrap confidence intervals...")
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    predictions = []
    
    # Generate multiple predictions with dropout enabled
    model.train()  # Set to train mode to enable dropout
    for _ in range(n_bootstrap):
        with torch.no_grad():
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
            y_pred = y_scaler.inverse_transform(y_pred_scaled)
            predictions.append(y_pred)
    
    # Convert to array for easier calculations
    predictions_array = np.array(predictions).squeeze()
    
    # Calculate mean and confidence intervals
    mean_prediction = np.mean(predictions_array, axis=0)
    lower_bound = np.percentile(predictions_array, 5, axis=0)
    upper_bound = np.percentile(predictions_array, 95, axis=0)
    
    # Sort for better visualization
    sorted_indices = np.argsort(y_test.values.flatten())
    sorted_actual = y_test.values.flatten()[sorted_indices]
    sorted_mean = mean_prediction[sorted_indices]
    sorted_lower = lower_bound[sorted_indices]
    sorted_upper = upper_bound[sorted_indices]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(range(len(sorted_actual)), sorted_lower, sorted_upper, 
                     alpha=0.3, label='90% Confidence Interval')
    plt.plot(range(len(sorted_actual)), sorted_mean, 'b-', label='Mean Prediction')
    plt.plot(range(len(sorted_actual)), sorted_actual, 'ro', markersize=3, label='Actual Values')
    
    plt.xlabel('Sorted Test Sample Index')
    plt.ylabel('dGmix Value')
    plt.title('Prediction with 90% Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save to both directories
    plt.savefig(f'{ff_graphs_dir}/confidence_intervals.png', dpi=300)
    plt.savefig(f'{graphs_dir}/confidence_intervals.png', dpi=300)
    logger.info(f"Confidence Intervals plot saved to {ff_graphs_dir}/confidence_intervals.png")
    plt.close()
    
    return mean_prediction, lower_bound, upper_bound

def save_model_info(model, feature_scaler, y_scaler, metrics, X_train_shape):
    """Save model and related information"""
    model_info = {
        'model_state': model.state_dict(),
        'scaler_state': {
            'feature_scaler_center': feature_scaler.center_,
            'feature_scaler_scale': feature_scaler.scale_,
            'target_scaler_mean': y_scaler.mean_,
            'target_scaler_scale': y_scaler.scale_
        },
        'metrics': {
            'mse': metrics[0],
            'rmse': metrics[1],
            'mae': metrics[2],
            'r2': metrics[3]
        },
        'model_params': {
            'input_features': X_train_shape[1]
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'device_info': {
            'platform': 'Apple Silicon' if torch.backends.mps.is_available() else 'CPU',
            'mps_available': torch.backends.mps.is_available()
        }
    }
    
    # Use memory-efficient saving
    buffer = io.BytesIO()
    torch.save(model_info, buffer, _use_new_zipfile_serialization=True)
    buffer.seek(0)
    
    # Save to both locations
    model_path = os.path.join(ff_models_dir, 'feedforward_model.pt')
    with open(model_path, 'wb') as f:
        f.write(buffer.read())
    
    # Also save to the original location for compatibility
    orig_model_path = '/Users/ca5/Desktop/qnn_fnl/models/feedforward_model.pt'
    os.makedirs(os.path.dirname(orig_model_path), exist_ok=True)
    with open(orig_model_path, 'wb') as f:
        buffer.seek(0)
        f.write(buffer.read())
    
    logger.info(f"Model saved to {model_path} and {orig_model_path}")

def main():
    try:
        # Load and preprocess data
        df, target_column = load_data("/Users/ca5/Desktop/qnn_fnl/data_filtered-1.csv")
        
        # Split and scale data
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_train, y_test, feature_scaler, y_scaler = preprocess_data(df, target_column)
        
        # Initialize model
        input_dim = X_train_scaled.shape[1]
        model = FeedforwardNN(input_dim=input_dim)
        logger.info(f"Initialized FeedforwardNN with {input_dim} input features")
        
        # Train model
        model, train_losses, val_losses = train_model(
            model, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            epochs=25, batch_size=32, patience=15
        )
        
        # Evaluate model
        y_pred, mse, rmse, mae, r2, errors = evaluate_model(model, X_test_scaled, y_test_scaled, y_test, y_scaler)
        
        # Create visualizations
        plot_actual_vs_predicted(y_test, y_pred, rmse)
        plot_qq_residuals(errors)
        mean_pred, lower_bound, upper_bound = plot_confidence_intervals(model, X_test_scaled, y_test, y_scaler)
        
        # Save model and info
        save_model_info(model, feature_scaler, y_scaler, (mse, rmse, mae, r2), X_train_scaled.shape)
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to both directories
        plt.savefig(f'{ff_graphs_dir}/learning_curves.png', dpi=300)
        plt.savefig(f'{graphs_dir}/learning_curves.png', dpi=300)
        logger.info(f"Learning curves saved to {ff_graphs_dir}/learning_curves.png")
        
        # Memory cleanup
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        logger.info("Feedforward Neural Network training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
