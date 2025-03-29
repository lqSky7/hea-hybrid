import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load the data
df = pd.read_csv('data_filtered-1.csv')
print(f"Original data shape: {df.shape}")

# Drop object columns
object_columns = df.select_dtypes(include=['object']).columns
print(f"Dropping object columns: {list(object_columns)}")
df = df.drop(columns=object_columns)

# Check for NaN values and handle them
if df.isna().sum().sum() > 0:
    print("Handling NaN values...")
    df = df.dropna()

# Separate features and target
X = df.drop(columns=['dGmix'])
y = df['dGmix']
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale to [0, 1] for quantum encoding
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train_scaled)
X_test_minmax = minmax_scaler.transform(X_test_scaled)

# Define number of qubits (limited by quantum processing capabilities)
n_features = X_train.shape[1]
n_qubits = min(n_features, 8)  # Using 8 qubits for computational feasibility
print(f"Using {n_qubits} qubits")

# Create qubits
qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]

# Convert classical data to quantum circuits
def convert_to_circuit(feature_vector):
    """Encode classical features into quantum circuit"""
    circuit = cirq.Circuit()
    
    # Encode features as rotation gates
    for i, feature in enumerate(feature_vector[:n_qubits]):
        # Encode features as rotation around X axis
        circuit.append(cirq.rx(np.pi * feature).on(qubits[i]))
    
    # Add some entanglement
    for i in range(n_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    
    return circuit

# Create quantum datasets
X_train_circuits = [convert_to_circuit(x) for x in X_train_minmax]
X_test_circuits = [convert_to_circuit(x) for x in X_test_minmax]

# Convert to TFQ tensors
X_train_tfq = tfq.convert_to_tensor(X_train_circuits)
X_test_tfq = tfq.convert_to_tensor(X_test_circuits)

# Create the parameterized quantum circuit
def create_quantum_model(n_layers=4):
    """Create a variational quantum circuit model"""
    # Create symbols for the parameters
    n_params = n_layers * (2 * n_qubits + (n_qubits - 1))
    params = sympy.symbols(f'theta(0:{n_params})')
    param_index = 0
    
    # Create the circuit with parameters
    circuit = cirq.Circuit()
    
    for l in range(n_layers):
        # Single qubit rotations
        for i in range(n_qubits):
            circuit.append(cirq.ry(params[param_index]).on(qubits[i]))
            param_index += 1
            circuit.append(cirq.rz(params[param_index]).on(qubits[i]))
            param_index += 1
        
        # Entangling gates
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
            param_index += 1
    
    # Prepare observables (readout operators)
    observables = [cirq.Z(q) for q in qubits]
    
    return circuit, observables, params

# Create the quantum circuit
circuit, observables, params = create_quantum_model()

# Build the hybrid quantum-classical model
# Classical pre-processing
classical_input = tf.keras.layers.Input(shape=(n_features,), name='classical_input')
dense1 = tf.keras.layers.Dense(64, activation='relu')(classical_input)
dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
dense3 = tf.keras.layers.Dense(n_qubits, activation='tanh')(dense2)

# Quantum processing
quantum_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='quantum_input')
quantum_layer = tfq.layers.PQC(circuit, observables)(quantum_input)

# Combine classical and quantum components
combined = tf.keras.layers.Concatenate()([dense3, quantum_layer])
output = tf.keras.layers.Dense(1)(combined)

# Create the hybrid model
hybrid_model = tf.keras.models.Model(
    inputs=[classical_input, quantum_input], 
    outputs=output
)

# Compile the model with advanced optimization settings
hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss='mse',
    metrics=['mae']
)

# Print model summary
hybrid_model.summary()

# Train the model
print("Training hybrid quantum-classical model...")
history = hybrid_model.fit(
    [X_train_scaled, X_train_tfq],
    y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
    ],
    verbose=1
)

# Evaluate the model
test_results = hybrid_model.evaluate([X_test_scaled, X_test_tfq], y_test)
print(f"Test Loss: {test_results[0]}, Test MAE: {test_results[1]}")

# Make predictions
predictions = hybrid_model.predict([X_test_scaled, X_test_tfq])

# Calculate additional metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"MSE: {mse}, R²: {r2}")

# Create output directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Plot results
plt.figure(figsize=(15, 5))

# Plot training & validation loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot predictions vs actual
plt.subplot(1, 3, 2)
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'Predictions vs True Values (R² = {r2:.3f})')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

# Plot error distribution
plt.subplot(1, 3, 3)
errors = y_test - predictions.flatten()
plt.hist(errors, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title(f'Error Distribution (MSE = {mse:.3f})')

# Save the plot
plt.tight_layout()
plt.savefig('results/quantum_results.png')
plt.show()

# Save the model
hybrid_model.save('results/hybrid_quantum_model')

print("Complete! Model saved to 'results/hybrid_quantum_model'")
