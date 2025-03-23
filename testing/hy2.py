import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from datetime import datetime

np.seterr(divide='ignore', invalid='ignore')

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "qnn_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('hybrid_qnn_trainer')

# Load and explore data
logger.info("Loading dataset...")
data = pd.read_csv("/Users/ca5/Desktop/qnn_fnl/data_filtered-1.csv")
logger.info(f"Data shape: {data.shape}")
logger.info(f"Columns: {data.columns.tolist()}")
logger.info(f"Sample:\n {data.head()}")

# Check data types and identify categorical columns
logger.info("\nData types:")
logger.info(f"{data.dtypes}")

# Identify categorical columns (usually objects or strings)
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
logger.info(f"\nCategorical columns: {categorical_columns}")

# Prepare features and target
X = data.drop('dGmix', axis=1)
y = data['dGmix']

# Handle categorical columns
if categorical_columns:
    X = X.drop(columns=categorical_columns)
    logger.info(f"After dropping categorical columns: {X.shape} features")

# Add polynomial features (from deepn1.py)
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
if numeric_cols:
    logger.info("Adding polynomial features (degree=2)")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    logger.info(f"Features shape after polynomial transformation: {X_poly.shape}")
    X = X_poly
else:
    logger.warning("No numeric columns found for polynomial features")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logger.info("Data scaling completed")

# Check for target scaling
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
logger.info("Target scaling completed")

# Enhanced quantum circuit with more layers
n_qubits = min(8, X.shape[1])  # Adjust based on feature count
n_layers = 3  # Increase number of layers for better expressivity
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Amplitude encoding of normalized inputs
    for i in range(n_qubits):
        qml.RY(inputs[i % len(inputs)], wires=i)
    
    # Multiple layers of parameterized gates and entanglement
    for l in range(n_layers):
        # Rotation layer
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        
        # Entanglement layer - fully connected
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.CNOT(wires=[i, j])
    
    # Measure all qubits to extract more information
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Enhanced Hybrid Model with better classical components
class EnhancedHybridModel(nn.Module):
    def __init__(self, n_features, n_qubits, n_layers=3):
        super(EnhancedHybridModel, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Pre quantum network - deeper with batch normalization
        self.pre_net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )
        
        # Quantum circuit weights
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        
        # Post quantum network - deeper with batch normalization
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Pre-processing with classical network
        x = self.pre_net(x)
        
        # Quantum processing with batched output
        q_out = torch.zeros(batch_size, self.n_qubits)
        for i, inputs in enumerate(x):
            q_result = quantum_circuit(inputs.detach().numpy(), 
                                      self.q_weights.detach().numpy())
            q_out[i] = torch.tensor(q_result)
        
        # Post-processing
        out = self.post_net(q_out)
        return out

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled).reshape(-1, 1)  # Corrected FloatFloat to FloatTensor

# Initialize enhanced model
model = EnhancedHybridModel(X_train_scaled.shape[1], n_qubits, n_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Create Tensorboard directory
tb_dir = os.path.join(log_dir, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(tb_dir, exist_ok=True)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(tb_dir)

# Enhanced training loop with early stopping
epochs = 25
batch_size = 32  # Larger batch size for stability
losses = []
val_losses = []
best_val_loss = float('inf')
patience = 15
patience_counter = 0
best_model_path = os.path.join(os.path.dirname(__file__), "best_hybrid_model.pt")

logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    # Mini-batch training
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
    
    # Store losses
    avg_epoch_loss = epoch_loss / (len(X_train_tensor) // batch_size + 1)
    losses.append(avg_epoch_loss)
    val_losses.append(val_loss)
    
    # Tensorboard logging
    writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"Epoch {epoch+1}: New best model saved with validation loss: {best_val_loss:.6f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    if (epoch+1) % 5 == 0:
        logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}')

# Load best model
model.load_state_dict(torch.load(best_model_path))
logger.info(f"Loaded best model from {best_model_path}")

# Evaluation with more metrics
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()
    
    # Convert scaled predictions back to original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test.values, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test.values, y_pred)
    r2 = r2_score(y_test.values, y_pred)

logger.info(f'Test MSE: {mse:.4f}')
logger.info(f'Test RMSE: {rmse:.4f}')
logger.info(f'Test MAE: {mae:.4f}')
logger.info(f'Test R²: {r2:.4f}')

# New visualization code starts here

import glob
# Use an absolute path for saving graphs
graphs_dir = "/Users/ca5/Desktop/qnn_fnl/graphs"
os.makedirs(graphs_dir, exist_ok=True)
logger.info(f"Graphs will be saved to: {graphs_dir}")

# 1. Learning Curves Plot
fig1 = plt.figure(figsize=(10,6))
plt.plot(losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.savefig(os.path.join(graphs_dir, "learning_curves.png"), dpi=300)
plt.close(fig1)

# 2. Actual vs Predicted (Test Set)
fig2 = plt.figure(figsize=(8,8))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual dGmix')
plt.ylabel('Predicted dGmix')
plt.title(f'Actual vs Predicted (Test), R²: {r2:.4f}')
plt.savefig(os.path.join(graphs_dir, "actual_vs_predicted_test.png"), dpi=300)
plt.close(fig2)

# 3. Error Distribution Histogram
error_vals = y_pred.flatten() - y_test.values.flatten()
fig3 = plt.figure(figsize=(10,6))
plt.hist(error_vals, bins=30, alpha=0.7)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title(f'Error Distribution, RMSE: {rmse:.4f}')
plt.savefig(os.path.join(graphs_dir, "error_distribution.png"), dpi=300)
plt.close(fig3)

# 4. Residual Plot
fig4 = plt.figure(figsize=(10,6))
plt.scatter(y_pred.flatten(), error_vals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual (Actual-Predicted)')
plt.title('Residual Plot')
plt.savefig(os.path.join(graphs_dir, "residual_plot.png"), dpi=300)
plt.close(fig4)

# 5. 3D Surface Plot Using Two Top Features (if available)
if X_test_scaled.shape[1] >= 2:
    try:
        from mpl_toolkits.mplot3d import Axes3D
        # Compute absolute correlations for each feature
        importance = [np.abs(np.corrcoef(X_train[:, i], y_train_scaled)[0, 1])
                      for i in range(X_train.shape[1])]
        top_two_idx = np.argsort(importance)[-2:]
        x_range = np.linspace(np.min(X_test_scaled[:, top_two_idx[0]]),
                              np.max(X_test_scaled[:, top_two_idx[0]]), 50)
        y_range = np.linspace(np.min(X_test_scaled[:, top_two_idx[1]]),
                              np.max(X_test_scaled[:, top_two_idx[1]]), 50)
        xx, yy = np.meshgrid(x_range, y_range)
        # Prepare inputs: fill grid with mean for all features, update two dimensions
        grid = np.tile(np.mean(X_test_scaled, axis=0), (xx.size, 1))
        grid[:, top_two_idx[0]] = xx.ravel()
        grid[:, top_two_idx[1]] = yy.ravel()
        grid_tensor = torch.FloatTensor(grid)
        with torch.no_grad():
            grid_pred = model(grid_tensor).numpy()
        grid_pred = y_scaler.inverse_transform(grid_pred).reshape(xx.shape)
        fig5 = plt.figure(figsize=(10,8))
        ax = fig5.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xx, yy, grid_pred, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_xlabel(f'Feature {top_two_idx[0]}')
        ax.set_ylabel(f'Feature {top_two_idx[1]}')
        ax.set_zlabel('dGmix')
        ax.set_title('3D Surface Plot')
        fig5.colorbar(surf, shrink=0.5, aspect=10)
        plt.savefig(os.path.join(graphs_dir, "3d_surface_plot.png"), dpi=300)
        plt.close(fig5)
    except Exception as e:
        logger.warning(f"Failed to generate 3D surface plot: {e}")

# 6. Overview of All Saved Plots
png_files = glob.glob(os.path.join(graphs_dir, "*.png"))
fig6 = plt.figure(figsize=(12,12))
for i, file in enumerate(png_files):
    img = plt.imread(file)
    ax = fig6.add_subplot(2, int(np.ceil(len(png_files) / 2)), i + 1)
    ax.imshow(img)
    ax.set_title(os.path.basename(file))
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "overview_plots.png"), dpi=300)
plt.close(fig6)

logger.info("All graphs created and saved.")

# Enhanced visualization
plt.figure(figsize=(15, 10))

# Plot training and validation loss
plt.subplot(2, 2, 1)
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot actual vs predicted
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual dGmix')
plt.ylabel('Predicted dGmix')
plt.title(f'Prediction vs Actual (R²: {r2:.4f})')

# Plot error distribution
plt.subplot(2, 2, 3)
errors = y_pred.flatten() - y_test.values
plt.hist(errors, bins=25)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title(f'Error Distribution (RMSE: {rmse:.4f})')

# Plot learning curve
plt.subplot(2, 2, 4)
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred.flatten(), label='Predicted')
plt.xlabel('Test Sample Index')
plt.ylabel('dGmix Value')
plt.title('Prediction vs Actual Values')
plt.legend()

plt.tight_layout()
plt.savefig('/Users/ca5/Desktop/qnn_fnl/enhanced_qnn_results.png', dpi=300)
logger.info("Results visualization saved to '/Users/ca5/Desktop/qnn_fnl/enhanced_qnn_results.png'")

# Save model and metadata
model_info = {
    'model_state': model.state_dict(),
    'scaler_state': {
        'feature_scaler_mean': scaler.mean_,
        'feature_scaler_scale': scaler.scale_,
        'target_scaler_mean': y_scaler.mean_,
        'target_scaler_scale': y_scaler.scale_
    },
    'metrics': {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    },
    'model_params': {
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'n_features': X_train_scaled.shape[1]
    },
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

torch.save(model_info, '/Users/ca5/Desktop/qnn_fnl/enhanced_hybrid_qnn_model.pt')
logger.info("Enhanced model saved to '/Users/ca5/Desktop/qnn_fnl/enhanced_hybrid_qnn_model.pt'")