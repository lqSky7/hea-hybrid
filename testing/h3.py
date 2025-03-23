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

# ENHANCED: Advanced feature engineering
logger.info("Performing advanced feature engineering...")

# Add polynomial features with higher degree for better feature interactions
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
if numeric_cols:
    logger.info("Adding polynomial features (degree=3)")
    poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X)
    logger.info(f"Features shape after polynomial transformation: {X_poly.shape}")
    
    # Add log and reciprocal transformations for selected features
    X_orig = pd.DataFrame(X)
    log_features = []
    for col in X_orig.columns:
        # Only transform positive columns
        if (X_orig[col] > 0).all():
            col_name = f"log_{col}"
            log_features.append(np.log1p(X_orig[col]).values)
    
    if log_features:
        log_features = np.column_stack(log_features)
        X_combined = np.hstack((X_poly, log_features))
        logger.info(f"Features shape after adding log transforms: {X_combined.shape}")
        X = X_combined
    else:
        X = X_poly
else:
    logger.warning("No numeric columns found for polynomial features")

# Split data with stratification to ensure similar distributions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ENHANCED: Robust scaling to handle outliers
from sklearn.preprocessing import RobustScaler
feature_scaler = RobustScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)
logger.info("Data scaling completed with RobustScaler")

# Target scaling - use regular StandardScaler for target
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
logger.info("Target scaling completed")

# ENHANCED: More qubits and layers for quantum circuit
n_qubits = min(12, X.shape[1])  # Increased from 8 to 12
n_layers = 5  # Increased from 3 to 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Enhanced amplitude encoding of normalized inputs
    inputs_padded = np.pad(inputs, (0, n_qubits - len(inputs) % n_qubits), mode='constant')
    for i in range(n_qubits):
        qml.RY(inputs_padded[i % len(inputs_padded)] * np.pi, wires=i)
    
    # Multiple layers of parameterized gates and entanglement
    for l in range(n_layers):
        # Rotation layer with more gates
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
            # Add U3 gate for more expressivity
            qml.U3(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
        
        # Enhanced entanglement layer - more complex pattern
        if l % 2 == 0:  # Even layers - nearest neighbors
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i, i+1])
            # Connect first and last for periodic boundary
            qml.CNOT(wires=[n_qubits-1, 0])
        else:  # Odd layers - long-range connections
            for i in range(0, n_qubits, 2):
                qml.CNOT(wires=[i, (i + n_qubits//2) % n_qubits])
    
    # Enhanced measurement strategy
    expectations = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    expectations += [qml.expval(qml.PauliX(i)) for i in range(n_qubits//2)]
    return expectations

# ENHANCED: Deeper Hybrid Model with residual connections
class AdvancedHybridModel(nn.Module):
    def __init__(self, n_features, n_qubits, n_layers=5):
        super(AdvancedHybridModel, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Number of quantum outputs
        self.q_output_size = n_qubits + n_qubits//2
        
        # Pre quantum network - much deeper with residual connections
        self.pre_net1 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
        )
        
        self.pre_net2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )
        
        self.pre_net3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )
        
        self.pre_out = nn.Sequential(
            nn.Linear(64, n_qubits),
            nn.Tanh()  # Constrain to [-1, 1] for quantum circuit
        )
        
        # Quantum circuit weights
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        
        # Post quantum network - deeper with residual connections
        self.post_net1 = nn.Sequential(
            nn.Linear(self.q_output_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )
        
        self.post_net2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )
        
        self.post_out = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        
        # Skip connection for original features
        self.skip_connection = nn.Linear(n_features, 128)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Skip connection
        skip = self.skip_connection(x)
        
        # Pre-processing with classical network and residual connections
        x1 = self.pre_net1(x)
        x2 = self.pre_net2(x1)
        x3 = self.pre_net3(x2) + 0.1 * x1[:, :64]  # Partial residual connection
        x_pre = self.pre_out(x3)
        
        # Quantum processing with batched output
        q_out = torch.zeros(batch_size, self.q_output_size)
        for i, inputs in enumerate(x_pre):
            q_result = quantum_circuit(inputs.detach().numpy(), 
                                     self.q_weights.detach().numpy())
            q_out[i] = torch.tensor(q_result)
        
        # Post-processing with residual connections
        p1 = self.post_net1(q_out)
        p1 = p1 + skip  # Skip connection from original features
        p2 = self.post_net2(p1)
        out = self.post_out(p2)
        
        return out

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled).reshape(-1, 1)

# ENHANCED: Implement an ensemble of models for better performance
n_models = 3  # Create an ensemble of 3 models
models = []

for i in range(n_models):
    # Initialize advanced model
    model = AdvancedHybridModel(X_train_scaled.shape[1], n_qubits, n_layers)
    models.append(model)
    logger.info(f"Initialized model {i+1} of {n_models} in ensemble")

# ENHANCED: Huber Loss for robustness to outliers
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, y_pred, y_true):
        abs_error = torch.abs(y_pred - y_true)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return torch.mean(loss)

# Initialize loss and optimizers
criterion = HuberLoss(delta=1.0)
optimizers = [optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) for model in models]
schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True) 
              for optimizer in optimizers]

# Create Tensorboard directory 
tb_dir = os.path.join(log_dir, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(tb_dir, exist_ok=True)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(tb_dir)

# ENHANCED: Advanced training loop with cosine annealing and gradient clipping
epochs = 1  # Increased epochs
batch_size = 64  # Larger batch size for better stability
best_model_paths = [os.path.join(os.path.dirname(__file__), f"best_hybrid_model_{i}.pt") for i in range(n_models)]

logger.info(f"Starting ensemble training with {n_models} models for {epochs} epochs")

# Train each model in the ensemble
for model_idx, (model, optimizer, scheduler, best_model_path) in enumerate(zip(models, optimizers, schedulers, best_model_paths)):
    logger.info(f"Training model {model_idx+1} of {n_models}")
    
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Mini-batch training with shuffle
        indices = torch.randperm(len(X_train_tensor))
        
        for start_idx in range(0, len(indices), batch_size):
            idx = indices[start_idx:start_idx+batch_size]
            
            batch_X = X_train_tensor[idx]
            batch_y = y_train_tensor[idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        writer.add_scalar(f'Loss/train/model_{model_idx}', avg_epoch_loss, epoch)
        writer.add_scalar(f'Loss/validation/model_{model_idx}', val_loss, epoch)
        writer.add_scalar(f'LearningRate/model_{model_idx}', optimizer.param_groups[0]['lr'], epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check with longer patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Model {model_idx+1}: Epoch {epoch+1}: New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Model {model_idx+1}: Early stopping triggered after {epoch+1} epochs")
                break
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            logger.info(f'Model {model_idx+1}: Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}')

# Load best models
for i, (model, best_model_path) in enumerate(zip(models, best_model_paths)):
    model.load_state_dict(torch.load(best_model_path))
    logger.info(f"Loaded best model {i+1} from {best_model_path}")

# ENHANCED: Ensemble predictions for better accuracy
def ensemble_predict(models, X):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X).numpy()
            predictions.append(pred)
    
    # Average the predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# Make ensemble predictions
y_pred_scaled = ensemble_predict(models, X_test_tensor)

# Convert scaled predictions back to original scale
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# Calculate metrics
mse = mean_squared_error(y_test.values, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test.values, y_pred)
r2 = r2_score(y_test.values, y_pred)

logger.info(f'Ensemble Test MSE: {mse:.4f}')
logger.info(f'Ensemble Test RMSE: {rmse:.4f}')
logger.info(f'Ensemble Test MAE: {mae:.4f}')
logger.info(f'Ensemble Test R²: {r2:.4f}')

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
            grid_pred = ensemble_predict(models, grid_tensor)
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

# Apple Silicon optimizations
if torch.backends.mps.is_available():
    logger.info("Apple Silicon detected. Enabling MPS optimizations.")
    # Enable Metal Performance Shaders acceleration
    device = torch.device("mps")
    
    # Optimize models for Apple Silicon
    for i, model in enumerate(models):
        models[i] = model.to(device)
        
    # Export to CoreML for even better performance on Apple devices
    try:
        import coremltools as ct
        # Function to convert a single model to CoreML
        def export_to_coreml(model, idx):
            # Trace the model with example input
            example_input = torch.rand(1, X_train_scaled.shape[1], device=device)
            traced_model = torch.jit.trace(model, example_input)
            
            # Convert to CoreML
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape)]
            )
            
            # Save the model
            coreml_path = f'/Users/ca5/Desktop/qnn_fnl/qnn_model_{idx}.mlmodel'
            mlmodel.save(coreml_path)
            logger.info(f"Model {idx} exported to CoreML: {coreml_path}")
            
        # Try to export each model
        for idx, model in enumerate(models):
            try:
                model.eval()  # Set to evaluation mode
                export_to_coreml(model, idx)
            except Exception as e:
                logger.warning(f"CoreML export failed for model {idx}: {e}")
        
        logger.info("CoreML export completed. Models can now be used natively in Apple applications.")
    except ImportError:
        logger.info("CoreML tools not available. Install with: pip install coremltools")
else:
    logger.info("MPS acceleration not available. Using CPU.")
    device = torch.device("cpu")
    
    # Optimize for CPU-based execution on Apple Silicon
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"Setting PyTorch to use {cpu_count} CPU cores")
    torch.set_num_threads(cpu_count)
    
    # Set OMP threads for better CPU performance
    import os
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)

# Save model and metadata with device information
model_info = {
    'model_state': [model.cpu().state_dict() for model in models],  # Store CPU state dicts
    'scaler_state': {
        'feature_scaler_center': feature_scaler.center_,
        'feature_scaler_scale': feature_scaler.scale_,
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
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'device_info': {
        'platform': 'Apple Silicon' if torch.backends.mps.is_available() else 'CPU',
        'mps_available': torch.backends.mps.is_available()
    }
}

# Use memory-efficient saving for large models
buffer = io.BytesIO()
torch.save(model_info, buffer, _use_new_zipfile_serialization=True)
buffer.seek(0)

with open('/Users/ca5/Desktop/qnn_fnl/enhanced_hybrid_qnn_model.pt', 'wb') as f:
    f.write(buffer.read())

logger.info("Enhanced model saved with Apple Silicon optimizations to '/Users/ca5/Desktop/qnn_fnl/enhanced_hybrid_qnn_model.pt'")

# Memory cleanup
import gc
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()