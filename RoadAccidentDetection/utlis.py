import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_accident_data(data_dir, img_size=(128, 128)):
    """
    Load and preprocess road accident image data
    
    Args:
        data_dir (str): Path to the main data directory
        img_size (tuple): Target size for images
    
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Define paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    # Initialize lists
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    # Load training data
    for label, class_name in enumerate(['no_accident', 'accident']):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            X_train.append(img)
            y_train.append(label)
    
    # Load validation data
    for label, class_name in enumerate(['no_accident', 'accident']):
        class_path = os.path.join(val_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            X_val.append(img)
            y_val.append(label)
    
    # Load test data
    for label, class_name in enumerate(['no_accident', 'accident']):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            X_test.append(img)
            y_test.append(label)
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    
    # Convert labels to arrays
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_accident_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create data generators with augmentation for training
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        batch_size (int): Batch size for training
    
    Returns:
        tuple: train_generator, val_generator
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val, batch_size=batch_size, shuffle=False
    )
    
    return train_generator, val_generator

def preprocess_image_for_prediction(image_path, img_size=(128, 128)):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path (str): Path to the image
        img_size (tuple): Target image size
    
    Returns:
        numpy array: Preprocessed image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    return img