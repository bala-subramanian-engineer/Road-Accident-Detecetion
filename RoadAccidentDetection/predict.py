import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import preprocess_image_for_prediction

class AccidentDetector:
    def __init__(self, model_path, img_size=(128, 128), threshold=0.5):
        """
        Initialize accident detector
        
        Args:
            model_path (str): Path to trained model
            img_size (tuple): Input image size expected by model
            threshold (float): Confidence threshold for accident detection
        """
        self.model = load_model(model_path)
        self.img_size = img_size
        self.threshold = threshold
    
    def predict(self, image_path):
        """
        Predict if an accident is detected in the image
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            dict: Prediction results with confidence score
        """
        # Preprocess image
        img = preprocess_image_for_prediction(image_path, self.img_size)
        
        # Make prediction
        prediction = self.model.predict(img)
        confidence = float(prediction[0][0])
        
        # Determine if accident is detected
        is_accident = confidence > self.threshold
        
        return {
            'accident_detected': bool(is_accident),
            'confidence': confidence,
            'threshold': self.threshold
        }
    
    def predict_batch(self, image_paths):
        """
        Predict accidents for multiple images
        
        Args:
            image_paths (list): List of paths to image files
        
        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'accident_detected': None,
                    'confidence': None
                })
        
        return results

def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Detect road accidents from images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to image to analyze or directory containing images')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Confidence threshold for accident detection')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128], 
                        help='Image size (height width)')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = AccidentDetector(args.model_path, tuple(args.img_size), args.threshold)
    
    # Check if input is a directory or single file
    if os.path.isdir(args.image_path):
        # Process all images in directory
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend([os.path.join(args.image_path, f) for f in os.listdir(args.image_path) 
                              if f.lower().endswith(ext.replace('*', ''))])
        
        results = detector.predict_batch(image_paths)
    else:
        # Process single image
        results = [detector.predict(args.image_path)]
        results[0]['image_path'] = args.image_path
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = []
            for result in results:
                json_result = {}
                for key, value in result.items():
                    if hasattr(value, 'item'):  # Convert numpy types
                        json_result[key] = value.item()
                    else:
                        json_result[key] = value
                json_results.append(json_result)
            json.dump(json_results, f, indent=2)
    
    # Print results
    for result in results:
        if 'error' in result:
            print(f"{result['image_path']}: Error - {result['error']}")
        else:
            status = "ACCIDENT DETECTED" if result['accident_detected'] else "No accident"
            print(f"{result['image_path']}: {status} (confidence: {result['confidence']:.4f})")

if __name__ == "__main__":
    main()