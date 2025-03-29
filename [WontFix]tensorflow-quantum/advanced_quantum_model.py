import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def prepare_data(file_path):
    """Load and preprocess data"""
    df = pd.read_csv(file_path)
    
    # Drop object columns
    object_columns = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=object_columns)
    
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop(columns=['dGmix'])
    y = df['dGmix']
    
    return X, y

def create_advanced_quantum_circuit(qubits, features=None, n_layers=4):
    """Create a more advanced quantum circuit with feature encoding"""
    circuit = cirq.Circuit()
    
    # Encode features if provided
    if features is not None:
        for i, feature in enumerate(features[:len(qubits)]):
            circuit.append(cirq.rx(np.pi * feature).on(qubits[i]))
            circuit.append(cirq.rz(np.pi * feature).on(qubits[i]))
    
    # Create variational parameters
    params = []
    symbols = []
    
    # Add variational layers
    for l in range(n_layers):
        for i in range(len(qubits)):
            # Rotation gates
            symbol_ry = sympy.Symbol(f'ry_{l}_{i}')
            symbol_rz = sympy.Symbol(f'rz_{l}_{i}')
            symbols.extend([symbol_ry, symbol_rz])
            
            circuit.append(cirq.ry(symbol_ry).on(qubits[i]))
            circuit.append(cirq.rz(symbol_rz).on(qubits[i]))
        
        # Add entangling layer - more complex entanglement pattern
        for i in range(len(qubits)):
            circuit.append(cirq.CNOT(qubits[i], qubits[(i + 1) % len(qubits)]))
            
        # Add non-local interactions in deeper layers
        if l >= n_layers // 2:
            for i in range(0, len(qubits), 2):
                if i + 2 < len(qubits):
                    circuit.append(cirq.CNOT(qubits[i], qubits[i + 2]))
    
    return circuit, symbols

def build_advanced_hybrid_model(n_features, n_qubits, readout_qubits=None):
    """Build a more sophisticated hybrid quantum-classical model"""
    if readout_qubits is None:
        readout_qubits = list(range(n_qubits))
    
    # Create qubits
    qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
    
    # Create quantum circuit
    circuit, symbols = create_advanced_quantum_circuit(qubits)
    
    # Define observables - use strategic combinations of qubits
    observables = [cirq.Z(qubits[i]) for i in readout_qubits]
    if len(readout_qubits) > 1:
        # Add parity observables for better feature extraction
        for i in range(len(readout_qubits) - 1):
            observables.append(cirq.Z(qubits[readout_qubits[i]]) @ cirq.Z(qubits[readout_qubits[i+1]]))
    
    # Classical pre-processing
    classical_input = tf.keras.layers.Input(shape=(n_features,), name='classical_input')
    x = tf.keras.layers.BatchNormalization()(classical_input)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    encoded_classical = tf.keras.layers.Dense(n_qubits, activation='tanh')(x)
    
    # Quantum processing
    quantum_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='quantum_input')
    quantum_layer = tfq.layers.PQC(circuit, observables)(quantum_input)
    
    # Post-processing
    combined = tf.keras.layers.Concatenate()([encoded_classical, quantum_layer])
    x = tf.keras.layers.Dense(16, activation='relu')(combined)
    output = tf.keras.layers.Dense(1)(x)
    
    # Create the model
    model = tf.keras.models.Model(
        inputs=[classical_input, quantum_input],
        outputs=output
    )
    
    return model

def main():
    # Load data
    X, y = prepare_data('data_filtered-1.csv')
    print(f"Data loaded: {X.shape} features, {y.shape} targets")
    
    # Configure model parameters
    n_features = X.shape[1]
    n_qubits = min(n_features, 8)  # Limit qubits for efficiency
    readout_qubits = list(range(min(4, n_qubits)))  # Only read from a subset of qubits
    
    # Create qubits
    qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
    
    # Implement k-fold cross-validation for better evaluation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold+1}/{n_splits}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Scale to [0, 1] for quantum encoding
        minmax_scaler = MinMaxScaler()
        X_train_minmax = minmax_scaler.fit_transform(X_train_scaled)
        X_test_minmax = minmax_scaler.transform(X_test_scaled)
        
        # Create quantum circuits for each data point
        X_train_circuits = [create_advanced_quantum_circuit(qubits, features=x)[0] for x in X_train_minmax]
        X_test_circuits = [create_advanced_quantum_circuit(qubits, features=x)[0] for x in X_test_minmax]
        
        # Convert to TFQ tensors
        X_train_tfq = tfq.convert_to_tensor(X_train_circuits)
        X_test_tfq = tfq.convert_to_tensor(X_test_circuits)
        
        # Build model
        model = build_advanced_hybrid_model(n_features, n_qubits, readout_qubits)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            [X_train_scaled, X_train_tfq],
            y_train,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=1
        )
        
        # Evaluate model
        y_pred = model.predict([X_test_scaled, X_test_tfq])
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Fold {fold+1} - MSE: {mse:.4f}, R²: {r2:.4f}")
        fold_results.append((mse, r2))
        
        # Save fold model
        os.makedirs('results', exist_ok=True)
        model.save(f'results/quantum_model_fold_{fold+1}')
    
    # Calculate average performance
    avg_mse = np.mean([res[0] for res in fold_results])
    avg_r2 = np.mean([res[1] for res in fold_results])
    print(f"\nAverage performance - MSE: {avg_mse:.4f}, R²: {avg_r2:.4f}")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    
    # Standardize all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Scale to [0, 1] for quantum encoding
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X_scaled)
    
    # Create quantum circuits
    X_circuits = [create_advanced_quantum_circuit(qubits, features=x)[0] for x in X_minmax]
    X_tfq = tfq.convert_to_tensor(X_circuits)
    
    # Build final model
    final_model = build_advanced_hybrid_model(n_features, n_qubits, readout_qubits)
    
    # Compile model
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    final_history = final_model.fit(
        [X_scaled, X_tfq],
        y,
        batch_size=32,
        epochs=100,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8)
        ],
        verbose=1
    )
    
    # Save final model
    final_model.save('results/final_quantum_model')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(final_history.history['loss'])
    plt.plot(final_history.history['val_loss'])
    plt.title('Final Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('results/final_model_training.png')
    
    print("Training complete! Final model saved to 'results/final_quantum_model'")

if __name__ == "__main__":
    main()
