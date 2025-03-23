# /Users/ca5/Desktop/qnn_fnl/train_qnn.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

# Load and explore data
data = pd.read_csv("/Users/ca5/Desktop/qnn_fnl/data_filtered-1.csv")
print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())
print("Sample:\n", data.head())

# Prepare features and target
X = data.drop('dGmix', axis=1)
y = data['dGmix']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define quantum device and circuit
n_qubits = min(8, X.shape[1])  # Adjust based on feature count
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode input data
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # First rotation layer
    for i in range(n_qubits):
        qml.RY(weights[0, i], wires=i)
    
    # Entanglement layer
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
    
    # Second rotation layer
    for i in range(n_qubits):
        qml.RY(weights[1, i], wires=i)
        
    # Return expectation value
    return qml.expval(qml.PauliZ(0))

# Hybrid Quantum-Classical Model
class HybridModel(nn.Module):
    def __init__(self, n_features, n_qubits):
        super(HybridModel, self).__init__()
        self.n_qubits = n_qubits
        self.pre_net = nn.Linear(n_features, n_qubits)
        self.q_weights = nn.Parameter(torch.randn(2, n_qubits))
        self.post_net = nn.Linear(1, 1)
    
    def forward(self, x):
        # Pre-processing
        x = torch.tanh(self.pre_net(x))
        
        # Quantum processing
        q_out = torch.zeros(x.shape[0])
        for i, inputs in enumerate(x):
            q_result = quantum_circuit(inputs.detach().numpy(), 
                                       self.q_weights.detach().numpy())
            q_out[i] = q_result
        
        # Post-processing
        out = self.post_net(q_out.unsqueeze(1))
        return out

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

# Initialize model
model = HybridModel(X_train_scaled.shape[1], n_qubits)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 50
batch_size = 16
losses = []

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
    
    losses.append(epoch_loss)
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Test MSE: {mse:.4f}')
print(f'Test R²: {r2:.4f}')

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual dGmix')
plt.ylabel('Predicted dGmix')
plt.title('Prediction vs Actual')

plt.tight_layout()
plt.savefig('/Users/ca5/Desktop/qnn_fnl/qnn_results.png')
plt.show()

# Save model
torch.save(model.state_dict(), '/Users/ca5/Desktop/qnn_fnl/hybrid_qnn_model.pt')
print("Model saved to '/Users/ca5/Desktop/qnn_fnl/hybrid_qnn_model.pt'")