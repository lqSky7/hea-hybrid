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
import io

np.seterr(divide='ignore', invalid='ignore')

# Force CPU usage only
device = torch.device("cpu")
print(f"Using {device} device.")

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

logger.info("Loading dataset...")
data = pd.read_csv("/Users/ca5/Desktop/qnn_fnl/data_filtered-1.csv")
logger.info(f"Data shape: {data.shape}")
logger.info(f"Columns: {data.columns.tolist()}")
logger.info(f"Sample:\n {data.head()}")

logger.info("\nData types:")
logger.info(f"{data.dtypes}")

categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
logger.info(f"\nCategorical columns: {categorical_columns}")

X = data.drop('dGmix', axis=1)
y = data['dGmix']

if categorical_columns:
    X = X.drop(columns=categorical_columns)
    logger.info(f"After dropping categorical columns: {X.shape} features")

logger.info("Performing advanced feature engineering...")

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
if numeric_cols:
    logger.info("Adding polynomial features (degree=3)")
    poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X)
    logger.info(f"Features shape after polynomial transformation: {X_poly.shape}")

    X_orig = pd.DataFrame(X)
    log_features = []
    for col in X_orig.columns:

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

from sklearn.preprocessing import RobustScaler
feature_scaler = RobustScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)
logger.info("Data scaling completed with RobustScaler")

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
logger.info("Target scaling completed")

n_qubits = min(12, X.shape[1])
n_layers = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):

    inputs_padded = np.pad(inputs, (0, n_qubits - len(inputs) % n_qubits), mode='constant')
    for i in range(n_qubits):
        qml.RY(inputs_padded[i % len(inputs_padded)] * np.pi, wires=i)

    for l in range(n_layers):

        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)

            qml.U3(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

        if l % 2 == 0:
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i, i+1])

            qml.CNOT(wires=[n_qubits-1, 0])
        else:
            for i in range(0, n_qubits, 2):
                qml.CNOT(wires=[i, (i + n_qubits//2) % n_qubits])

    expectations = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    expectations += [qml.expval(qml.PauliX(i)) for i in range(n_qubits//2)]
    return expectations

class AdvancedHybridModel(nn.Module):
    def __init__(self, n_features, n_qubits, n_layers=5):
        super(AdvancedHybridModel, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.q_output_size = n_qubits + n_qubits//2

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
            nn.Tanh()
        )

        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

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

        self.skip_connection = nn.Linear(n_features, 128)
    
    def forward(self, x):
        batch_size = x.shape[0]

        x = x.to(device)

        skip = self.skip_connection(x)

        x1 = self.pre_net1(x)
        x2 = self.pre_net2(x1)
        x3 = self.pre_net3(x2) + 0.1 * x1[:, :64]
        x_pre = self.pre_out(x3)

        q_out = torch.zeros(batch_size, self.q_output_size, device=device)
        for i, inputs in enumerate(x_pre):
            q_result = quantum_circuit(inputs.detach().cpu().numpy(), 
                                     self.q_weights.detach().cpu().numpy())
            q_result_array = np.array(q_result)  # Convert to numpy array
            q_out[i] = torch.tensor(q_result_array, dtype=torch.float, device=device)

        p1 = self.post_net1(q_out)
        p1 = p1 + skip
        p2 = self.post_net2(p1)
        out = self.post_out(p2)
        
        return out

X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train_scaled).reshape(-1, 1).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test_scaled).reshape(-1, 1).to(device)

# Train only one model instead of an ensemble
n_models = 1
model = AdvancedHybridModel(X_train_scaled.shape[1], n_qubits, n_layers).to(device)
logger.info(f"Initialized model on {device}")

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

criterion = HuberLoss(delta=1.0)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

tb_dir = os.path.join(log_dir, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(tb_dir, exist_ok=True)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(tb_dir)

epochs = 1
batch_size = 64
best_model_path = os.path.join(os.path.dirname(__file__), "best_hybrid_model.pt")

logger.info(f"Starting training for {epochs} epochs")

losses = []
val_losses = []
best_val_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    indices = torch.randperm(len(X_train_tensor))
    
    for start_idx in range(0, len(indices), batch_size):
        idx = indices[start_idx:start_idx+batch_size]
        
        batch_X = X_train_tensor[idx]
        batch_y = y_train_tensor[idx]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()

    avg_epoch_loss = epoch_loss / (len(X_train_tensor) // batch_size + 1)
    losses.append(avg_epoch_loss)
    val_losses.append(val_loss)

    writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

    scheduler.step(val_loss)

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
    
    if (epoch+1) % 5 == 0 or epoch == 0:
        logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}')

# Load the best model
model.load_state_dict(torch.load(best_model_path))
logger.info(f"Loaded best model from {best_model_path}")

def predict(model, X):
    model.eval()
    with torch.no_grad():
        if X.device != next(model.parameters()).device:
            X = X.to(next(model.parameters()).device)
        pred = model(X).cpu().numpy()
    return pred

y_pred_scaled = predict(model, X_test_tensor)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test.values, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test.values, y_pred)
r2 = r2_score(y_test.values, y_pred)

logger.info(f'Test MSE: {mse:.4f}')
logger.info(f'Test RMSE: {rmse:.4f}')
logger.info(f'Test MAE: {mae:.4f}')
logger.info(f'Test R²: {r2:.4f}')

# Create visualizations
import glob

graphs_dir = "/Users/ca5/Desktop/qnn_fnl/graphs"
os.makedirs(graphs_dir, exist_ok=True)
logger.info(f"Graphs will be saved to: {graphs_dir}")

fig1 = plt.figure(figsize=(10,6))
plt.plot(losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.savefig(os.path.join(graphs_dir, "learning_curves.png"), dpi=300)
plt.close(fig1)

fig2 = plt.figure(figsize=(8,8))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual dGmix')
plt.ylabel('Predicted dGmix')
plt.title(f'Actual vs Predicted (Test), R²: {r2:.4f}')
plt.savefig(os.path.join(graphs_dir, "actual_vs_predicted_test.png"), dpi=300)
plt.close(fig2)

error_vals = y_pred.flatten() - y_test.values.flatten()
fig3 = plt.figure(figsize=(10,6))
plt.hist(error_vals, bins=30, alpha=0.7)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title(f'Error Distribution, RMSE: {rmse:.4f}')
plt.savefig(os.path.join(graphs_dir, "error_distribution.png"), dpi=300)
plt.close(fig3)

fig4 = plt.figure(figsize=(10,6))
plt.scatter(y_pred.flatten(), error_vals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual (Actual-Predicted)')
plt.title('Residual Plot')
plt.savefig(os.path.join(graphs_dir, "residual_plot.png"), dpi=300)
plt.close(fig4)

# Save the final model
model_info = {
    'model_state': model.cpu().state_dict(),
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
        'platform': 'CPU',
        'mps_available': False
    }
}

buffer = io.BytesIO()
torch.save(model_info, buffer, _use_new_zipfile_serialization=True)
buffer.seek(0)

with open('/Users/ca5/Desktop/qnn_fnl/hybrid_qnn_model.pt', 'wb') as f:
    f.write(buffer.read())

logger.info("Model saved to '/Users/ca5/Desktop/qnn_fnl/hybrid_qnn_model.pt'")

import gc
gc.collect()