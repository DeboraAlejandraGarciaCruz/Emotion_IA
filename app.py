import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def adjust_brightness(image, value):
    """Adjust the brightness of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def apply_transformation(image_path, transformation, brightness_value=0):
    """Apply the selected transformation to the image."""
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Could not load image")

    if transformation == "flip_horizontal":
        image = cv2.flip(image, 1)
    elif transformation == "flip_vertical":
        image = cv2.flip(image, 0)
    elif transformation == "adjust_brightness":
        image = adjust_brightness(image, brightness_value)
    
    return image

def analyze_face(image_path, transformation=None, brightness_value=0):
    try:
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Read and optionally transform image
        if transformation:
            image = apply_transformation(image_path, transformation, brightness_value)
        else:
            image = cv2.imread(image_path)

        if image is None:
            raise Exception("Could not load image")

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect facial landmarks
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise Exception("No face detected in the image")

        # Key points for visualization
        key_points = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130]
        height, width = gray_image.shape

        # Create a new figure for each analysis
        plt.clf()
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(gray_image, cmap='gray')

        # Plot facial landmarks
        for point_idx in key_points:
            landmark = results.multi_face_landmarks[0].landmark[point_idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            plt.plot(x, y, 'rx')

        # Save plot to memory
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Convert to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

@app.route('/')
def home():
    # Get list of images in upload folder
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(filename)
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if we're analyzing an existing file
        if 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': f'File not found: {filename}'}), 404
            
        # Check if we're uploading a new file
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        
        else:
            return jsonify({'error': 'No file provided'}), 400

        # Get transformation type and brightness value
        transformation = request.form.get('transformation', None)
        brightness_value = int(request.form.get('brightness', 0))  # Default: no change in brightness

        # Analyze the image
        result_image = analyze_face(filepath, transformation, brightness_value)
        
        return jsonify({
            'success': True,
            'image': result_image
        })

    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
