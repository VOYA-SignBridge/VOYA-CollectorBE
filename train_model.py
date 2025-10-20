"""
Simple training script for sign language recognition
Dataset format: memmap X (N, 60, 226), y (N,)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

# Configuration
DATASET_PATH = "dataset/processed/memmap"
MODEL_SAVE_PATH = "models"
EPOCHS = 50
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2

def load_dataset():
    """Load memmap dataset"""
    print("Loading dataset...")
    
    # Load metadata
    meta_path = os.path.join(DATASET_PATH, "dataset_meta.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    print(f"Dataset info: {meta['total_samples']} samples, shape ({meta['sequence_length']}, {meta['feature_dim']})")
    
    # Load memmap arrays
    X_path = os.path.join(DATASET_PATH, "dataset_X.dat")
    y_path = os.path.join(DATASET_PATH, "dataset_y.dat")
    
    X = np.memmap(X_path, dtype=np.float32, mode='r', 
                  shape=(meta['total_samples'], meta['sequence_length'], meta['feature_dim']))
    y = np.memmap(y_path, dtype=np.int32, mode='r', shape=(meta['total_samples'],))
    
    # Convert to regular arrays for easier manipulation
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded X: {X.shape}, y: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, meta

def create_model(sequence_length, feature_dim, num_classes):
    """Create LSTM-based model for sequence classification"""
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(sequence_length, feature_dim)),
        
        # Feature extraction
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Temporal modeling
        layers.LSTM(64, return_sequences=True, dropout=0.3),
        layers.LSTM(32, dropout=0.3),
        
        # Classification head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Main training function"""
    # Load data
    X, y, meta = load_dataset()
    
    # Get unique classes
    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create model
    model = create_model(meta['sequence_length'], meta['feature_dim'], num_classes)
    model.summary()
    
    # Callbacks
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Predictions and metrics
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))
    
    # Save final model
    model.save(os.path.join(MODEL_SAVE_PATH, 'final_model.h5'))
    
    # Save training info
    training_info = {
        'dataset_meta': meta,
        'num_classes': num_classes,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'final_test_accuracy': float(test_acc),
        'final_test_loss': float(test_loss),
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy']))
    }
    
    with open(os.path.join(MODEL_SAVE_PATH, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\nTraining completed! Models saved to {MODEL_SAVE_PATH}/")
    return model, history

def predict_sample(model_path, X_sample):
    """Predict single sample"""
    model = tf.keras.models.load_model(model_path)
    
    # Ensure correct shape (1, 60, 226)
    if X_sample.ndim == 2:
        X_sample = X_sample.reshape(1, *X_sample.shape)
    
    prediction = model.predict(X_sample)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        print("Please run: curl -X POST http://localhost:8000/api/dataset/export")
        exit(1)
    
    # Train model
    model, history = train_model()
    
    print("\nTraining summary:")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")