import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from model import create_accident_detection_model, compile_model
from utils import load_accident_data, create_accident_data_generators

def train_accident_model(data_dir, model_save_path, epochs=50, batch_size=32, img_size=(128, 128)):
    """
    Train the road accident detection model
    
    Args:
        data_dir (str): Path to the data directory
        model_save_path (str): Path to save the trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (tuple): Target image size
    """
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_accident_data(data_dir, img_size)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create data generators
    train_generator, val_generator = create_accident_data_generators(
        X_train, y_train, X_val, y_val, batch_size
    )
    
    # Create model
    input_shape = (img_size[0], img_size[1], 3)
    model = create_accident_detection_model(input_shape)
    model = compile_model(model)
    
    # Display model summary
    model.summary()
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger('training_log.csv')
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train road accident detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='accident_detection_model.h5', 
                        help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128], 
                        help='Image size (height width)')
    
    args = parser.parse_args()
    
    # Train model
    train_accident_model(
        data_dir=args.data_dir,
        model_save_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size)
    )