import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_face(image_path):
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not load image")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise Exception("No face detected in the image")

        key_points = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130]
        height, width = gray_image.shape

        transformations = [
            ("Original", gray_image),
            ("Horizontally Flipped", cv2.flip(gray_image, 1)),
            ("Brightened", cv2.convertScaleAbs(gray_image, alpha=1.2, beta=50)),
            ("Upside Down", cv2.flip(gray_image, 0))
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for ax, (title, img) in zip(axes, transformations):
            ax.imshow(img, cmap='gray')
            for point_idx in key_points:
                landmark = results.multi_face_landmarks[0].landmark[point_idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                if title == "Horizontally Flipped":
                    x = width - x
                elif title == "Upside Down":
                    y = height - y
                ax.plot(x, y, 'rx')
            ax.set_title(title)
            ax.axis('off')

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        landmarks = results.multi_face_landmarks[0].landmark
        emotion = estimate_emotion(landmarks, width, height)

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64, emotion

    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

def estimate_emotion(landmarks, width, height):
    try:
        def dist(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        top_mouth = landmarks[0]
        bottom_mouth = landmarks[17]

        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]

        left_eyebrow = landmarks[65]
        right_eyebrow = landmarks[295]
        center_brow = landmarks[9]

        mouth_width = dist(left_mouth, right_mouth)
        mouth_height = dist(top_mouth, bottom_mouth)
        eye_open_left = dist(left_eye_top, left_eye_bottom)
        eye_open_right = dist(right_eye_top, right_eye_bottom)
        brow_distance = dist(left_eyebrow, right_eyebrow)
        brow_to_center = dist(center_brow, top_mouth)

        print("Mouth Height:", mouth_height, "Width:", mouth_width)
        print("Eye Open L:", eye_open_left, "R:", eye_open_right)
        print("Brow Dist:", brow_distance, "Brow to Mouth:", brow_to_center)

        if mouth_height > 0.06 and eye_open_left > 0.05:
            return "Sorpresa"
        elif mouth_height > 0.045 and mouth_width > 0.35:
            return "Alegría"
        elif eye_open_left < 0.015 and eye_open_right < 0.015:
            return "Tristeza"
        elif brow_distance < 0.10 and brow_to_center < 0.06:
            return "Enojo"
        else:
            return "Neutral"

    except Exception as e:
        print(f"Error in estimate_emotion: {e}")
        return "Desconocida"

@app.route('/')
def home():
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(filename)
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': f'File not found: {filename}'}), 404

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

        result_image, emotion = analyze_face(filepath)

        return jsonify({
            'success': True,
            'image': result_image,
            'emotion': emotion
        })

    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
