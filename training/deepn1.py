import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import logging
import datetime

log_dir = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "training.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('deep_nn_trainer')

def load_and_check_data(file_path):
    
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully with shape: {df.shape}")

        logger.info(f"Column names: {df.columns.tolist()}")
        logger.info(f"Data sample:\n{df.head()}")
        logger.info(f"Data types:\n{df.dtypes}")
        logger.info(f"Missing values summary:\n{df.isnull().sum()}")
        logger.info(f"Data description:\n{df.describe()}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df, target_column):
    
    logger.info("Starting data preprocessing with enhanced feature engineering")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Numeric columns: {numeric_cols}")
    logger.info(f"Categorical columns: {categorical_cols}")

    if categorical_cols:
        logger.info("Applying one-hot encoding to categorical columns")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        logger.info(f"One-hot encoded features shape: {X.shape}")

    if numeric_cols and len(numeric_cols) > 0:
        logger.info("Adding polynomial features")
        from sklearn.preprocessing import PolynomialFeatures

        X_numeric = X[numeric_cols]

        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(X_numeric)

        poly_feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]

        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

        X_without_numeric = X.drop(columns=numeric_cols)
        X = pd.concat([X_without_numeric, poly_df], axis=1)
        
        logger.info(f"Features shape after polynomial transformation: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Data scaling completed")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def build_model(input_dim, output_shape=1, is_classification=True):
    
    logger.info(f"Building improved model with input dimension {input_dim}")

    model = keras.Sequential([

        keras.layers.Dense(256, activation='relu', input_shape=(input_dim,),
                          kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation='relu',
                          kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(output_shape, 
                          activation='sigmoid' if is_classification and output_shape == 1 
                          else 'softmax' if is_classification 
                          else None)
    ])

    if is_classification:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
    else:

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.Huber(), 
            metrics=['mae', 'mse']
        )
    
    logger.info("Improved model built and compiled successfully")
    model.summary(print_fn=lambda x: logger.info(x))
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    
    logger.info(f"Training model with {epochs} epochs and batch size {batch_size}")

    os.makedirs("logs/fit", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        min_delta=0.001
    )

    checkpoint_path = "checkpoints/model_{epoch:02d}_{val_loss:.2f}.keras"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.01)
    
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, 
        min_lr=0.00001
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard_callback, early_stopping, checkpoint_callback, reduce_lr],
        verbose=1
    )
    
    logger.info("Model training completed")
    return history

def evaluate_model(model, X_test, y_test, is_classification=True):
    
    logger.info("Evaluating model performance")

    evaluation_results = model.evaluate(X_test, y_test)

    if isinstance(evaluation_results, list):
        if len(evaluation_results) >= 2:
            loss = evaluation_results[0]
            mae = evaluation_results[1]
        else:
            logger.error(f"Unexpected evaluation results format: {evaluation_results}")
            loss, mae = 0, 0
    else:
        loss, mae = evaluation_results, 0
    
    metric_name = 'accuracy' if is_classification else 'mae'
    logger.info(f"Test loss: {loss:.4f}")
    logger.info(f"Test {metric_name}: {mae:.4f}")

    print(f"\n{'Accuracy' if is_classification else 'Mean Absolute Error'} of trained model: {mae:.4f}")

    y_pred = model.predict(X_test)

    if is_classification:
        from sklearn.metrics import classification_report, confusion_matrix
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
            if hasattr(y_test, 'values'):
                y_true = y_test.values
            else:
                y_true = y_test
        else:
            y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
            y_true = y_test
            
        logger.info(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred_classes)}")
        logger.info(f"Classification report:\n{classification_report(y_true, y_pred_classes)}")
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

        print(f"R² Score: {r2:.4f}")
    
    return y_pred, mae

def plot_history(history, is_classification=True):
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    metric = 'accuracy' if is_classification else 'mae'
    plt.plot(history.history[metric], label=f'Training {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.title(metric.capitalize())
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    logger.info("Training history plot saved as 'training_history.png'")

def main():
    try:
        logger.info("Starting deep neural network training process")

        file_path = "/Users/ca5/Desktop/qnn_fnl/data_filtered-1.csv"  
        df = load_and_check_data(file_path)

        target_column = "dGmix"

        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in dataset.")
            logger.info(f"Available columns: {df.columns.tolist()}")
            return

        if not pd.api.types.is_numeric_dtype(df[target_column]):
            logger.warning(f"Target column '{target_column}' is not numeric. Converting to numeric.")
            try:
                df[target_column] = pd.to_numeric(df[target_column])
            except:
                logger.error(f"Failed to convert target column '{target_column}' to numeric.")
                return

        is_classification = False

        X_train, X_test, y_train, y_test, scaler = preprocess_data(df, target_column)

        input_dim = X_train.shape[1]
        output_shape = 1
        model = build_model(input_dim, output_shape, is_classification)

        history = train_model(model, X_train, y_train, X_test, y_test, epochs=100)

        y_pred, model_accuracy = evaluate_model(model, X_test, y_test, is_classification)

        plot_history(history, is_classification)

        model_path = "final_model.keras"
        model.save(model_path)
        logger.info(f"Final model saved to {model_path} using native Keras format")

        metric_name = 'Accuracy' if is_classification else 'Mean Absolute Error'
        print(f"\nFinal model {metric_name}: {model_accuracy:.4f}")
        print(f"Model saved to: {os.path.abspath(model_path)}")
        
        logger.info("Deep neural network training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
