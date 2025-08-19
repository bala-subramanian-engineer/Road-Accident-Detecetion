from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'accident_detection_model.h5'
IMG_SIZE = (128, 128)
THRESHOLD = 0.5

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
def load_detection_model():
    global model
    try:
        model = load_model(MODEL_PATH)
        print("Accident detection model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

load_detection_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/detect', methods=['POST'])
def detect_accident():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Preprocess image
            img = preprocess_image(filepath)
            
            # Make prediction
            prediction = model.predict(img)
            confidence = float(prediction[0][0])
            accident_detected = confidence > THRESHOLD
            
            # Prepare response
            response = {
                'accident_detected': bool(accident_detected),
                'confidence': confidence,
                'threshold': THRESHOLD,
                'alert_needed': accident_detected  # For integration with alert systems
            }
            
            # Clean up
            os.remove(filepath)
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/batch_detect', methods=['POST'])
def batch_detect():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            try:
                # Preprocess image
                img = preprocess_image(filepath)
                
                # Make prediction
                prediction = model.predict(img)
                confidence = float(prediction[0][0])
                accident_detected = confidence > THRESHOLD
                
                results.append({
                    'filename': file.filename,
                    'accident_detected': bool(accident_detected),
                    'confidence': confidence,
                    'threshold': THRESHOLD,
                    'alert_needed': accident_detected
                })
                
                # Clean up
                os.remove(filepath)
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
    
    return jsonify({'results': results})

@app.route('/health', methods=['GET'])
def health_check():
    if model is not None:
        return jsonify({'status': 'healthy'})
    else:
        return jsonify({'status': 'unhealthy'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)