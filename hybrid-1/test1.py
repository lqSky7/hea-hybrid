# /Users/ca5/Desktop/qnn_fnl/hybrid_nn_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import os
import logging
from datetime import datetime
from scipy import stats
warnings.filterwarnings('ignore')

# Set up directories
hybrid_dir = '/Users/ca5/Desktop/qnn_fnl/hybrid'
graphs_dir = os.path.join(hybrid_dir, 'graphs')
models_dir = os.path.join(hybrid_dir, 'models')
log_dir = os.path.join(hybrid_dir, 'logs')

# Create directories if they don't exist
for directory in [graphs_dir, models_dir, log_dir]:
    os.makedirs(directory, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"hybrid_nn_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('hybrid_nn_trainer')

# Load the dataset
logger.info("Loading dataset...")
data = pd.read_csv('/Users/ca5/Desktop/qnn_fnl/data_filtered-1.csv')

# Data exploration
logger.info("Dataset loaded, shape: %s", data.shape)
logger.info("Dataset columns: %s", data.columns.tolist())

# Drop object columns and handle NaN values
object_columns = data.select_dtypes(include=['object']).columns
data = data.drop(columns=object_columns)
logger.info("Dropped object columns: %s", object_columns.tolist())

# Check for NaN values and handle them
logger.info("NaN values in each column: %s", data.isna().sum().to_dict())
data = data.dropna()
logger.info("Shape after dropping NaN values: %s", data.shape)

# Separate features and target
X = data.drop(columns=['dGmix'])
y = data['dGmix']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info("Train set: %s, Test set: %s", X_train.shape, X_test.shape)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logger.info("Data scaling completed")

# Create traditional ML models for ensemble
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
svr_model = SVR(kernel='rbf')

# Train traditional ML models
logger.info("Training traditional ML models...")
rf_model.fit(X_train_scaled, y_train)
gb_model.fit(X_train_scaled, y_train)
svr_model.fit(X_train_scaled, y_train)

# Get predictions from traditional models for training set
rf_train_preds = rf_model.predict(X_train_scaled).reshape(-1, 1)
gb_train_preds = gb_model.predict(X_train_scaled).reshape(-1, 1)
svr_train_preds = svr_model.predict(X_train_scaled).reshape(-1, 1)

# Get predictions from traditional models for test set
rf_test_preds = rf_model.predict(X_test_scaled).reshape(-1, 1)
gb_test_preds = gb_model.predict(X_test_scaled).reshape(-1, 1)
svr_test_preds = svr_model.predict(X_test_scaled).reshape(-1, 1)

# Create enhanced feature sets by combining original features with model predictions
X_train_enhanced = np.hstack([X_train_scaled, rf_train_preds, gb_train_preds, svr_train_preds])
X_test_enhanced = np.hstack([X_test_scaled, rf_test_preds, gb_test_preds, svr_test_preds])

# Build a deep neural network
def build_hybrid_nn_model(input_dim):
    # Original features branch
    input_features = Input(shape=(input_dim,))
    x1 = Dense(128, activation='relu')(input_features)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(64, activation='relu')(x1)
    x1 = Dropout(0.2)(x1)
    
    # Model predictions branch
    input_preds = Input(shape=(3,))  # 3 model predictions
    x2 = Dense(16, activation='relu')(input_preds)
    
    # Combine branches
    combined = Concatenate()([x1, x2])
    
    # Final layers
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    output = Dense(1, activation='linear')(x)
    
    # Create model
    model = Model(inputs=[input_features, input_preds], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

# Split enhanced features
X_train_orig = X_train_scaled
X_train_preds = np.hstack([rf_train_preds, gb_train_preds, svr_train_preds])
X_test_orig = X_test_scaled
X_test_preds = np.hstack([rf_test_preds, gb_test_preds, svr_test_preds])

# Build and train hybrid model
hybrid_model = build_hybrid_nn_model(X_train_orig.shape[1])

# Callbacks for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model
logger.info("Training Hybrid Neural Network...")
history = hybrid_model.fit(
    [X_train_orig, X_train_preds], y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model
y_pred_nn = hybrid_model.predict([X_test_orig, X_test_preds])
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_gb = gb_model.predict(X_test_scaled)
y_pred_svr = svr_model.predict(X_test_scaled)

# Calculate metrics
logger.info("Model Evaluation Metrics:")
def print_metrics(name, predictions, actual):
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

rf_metrics = print_metrics("Random Forest", y_pred_rf, y_test)
gb_metrics = print_metrics("Gradient Boosting", y_pred_gb, y_test)
svr_metrics = print_metrics("SVR", y_pred_svr, y_test)
nn_metrics = print_metrics("Hybrid Neural Network", y_pred_nn.flatten(), y_test)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'training_history.png'))
logger.info(f"Training history plot saved to {os.path.join(graphs_dir, 'training_history.png')}")
plt.close()

# Function to plot actual vs predicted values
def plot_actual_vs_predicted(y_test, y_pred, metrics, title='Hybrid Neural Network'):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), np.min(y_pred))
    max_val = max(y_test.max(), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual dGmix')
    plt.ylabel('Predicted dGmix')
    plt.title(f'{title}: Actual vs Predicted Values (RMSE: {metrics["rmse"]:.4f}, R²: {metrics["r2"]:.4f})')
    
    # Add annotation with metrics
    plt.annotate(f'RMSE: {metrics["rmse"]:.4f}\nMAE: {metrics["mae"]:.4f}\nR²: {metrics["r2"]:.4f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'actual_vs_predicted_test.png'), dpi=300)
    logger.info(f"Actual vs Predicted plot saved to {os.path.join(graphs_dir, 'actual_vs_predicted_test.png')}")
    plt.close()

# Function to plot QQ plot of residuals
def plot_qq_residuals(y_test, y_pred):
    errors = y_test.values - y_pred.flatten()
    plt.figure(figsize=(10, 6))
    stats.probplot(errors, plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'qq_plot_residuals.png'), dpi=300)
    logger.info(f"Q-Q Plot of Residuals saved to {os.path.join(graphs_dir, 'qq_plot_residuals.png')}")
    plt.close()
    return errors

# Function to create confidence intervals using bootstrap
def plot_confidence_intervals(X_test_orig, X_test_preds, y_test, model, n_bootstrap=100):
    logger.info("Generating bootstrap confidence intervals...")
    predictions = []
    
    # Generate multiple predictions with dropout enabled (in train mode)
    model.trainable = False  # Disable training updates
    for _ in range(n_bootstrap):
        pred = model.predict([X_test_orig, X_test_preds], verbose=0)
        predictions.append(pred)
    
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
    plt.savefig(os.path.join(graphs_dir, 'confidence_intervals.png'), dpi=300)
    logger.info(f"Confidence Intervals plot saved to {os.path.join(graphs_dir, 'confidence_intervals.png')}")
    plt.close()
    
    return mean_prediction, lower_bound, upper_bound

# Generate the advanced visualization graphs
plot_actual_vs_predicted(y_test, y_pred_nn, nn_metrics)
errors = plot_qq_residuals(y_test, y_pred_nn)
mean_pred, lower_bound, upper_bound = plot_confidence_intervals(X_test_orig, X_test_preds, y_test, hybrid_model)

# Create a function to save the model
def save_hybrid_model(model, rf, gb, svr, scaler, metrics, output_dir=models_dir):
    # Save neural network
    model_path = os.path.join(output_dir, 'hybrid_nn_model.h5')
    model.save(model_path)
    
    # Save traditional models
    import joblib
    joblib.dump(rf, os.path.join(output_dir, 'rf_model.pkl'))
    joblib.dump(gb, os.path.join(output_dir, 'gb_model.pkl'))
    joblib.dump(svr, os.path.join(output_dir, 'svr_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # Save metrics
    import json
    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"All models saved to {output_dir}")

# Save the models
save_hybrid_model(hybrid_model, rf_model, gb_model, svr_model, scaler, nn_metrics)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'RF Importance': rf_model.feature_importances_
}).sort_values('RF Importance', ascending=False)

logger.info("Top 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    logger.info(f"{row['Feature']}: {row['RF Importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='RF Importance', y='Feature', data=feature_importance.head(15))
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'feature_importance.png'))
logger.info(f"Feature importance plot saved to {os.path.join(graphs_dir, 'feature_importance.png')}")
plt.close()

logger.info("Hybrid Neural Network implementation complete!")