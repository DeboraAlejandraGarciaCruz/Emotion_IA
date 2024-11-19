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
import copy
import random

app = Flask(__name__)

# Configuración del folder de uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Tamaño máximo de 16MB

# Asegurar que el folder de uploads existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_face(image_path):
    try:
        # Inicializamos MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Cargar la imagen
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("No se pudo cargar la imagen")

        # Convertimos a RGB y escala de grises
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectamos puntos faciales
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise Exception("No se detectó ningún rostro en la imagen")

        # Puntos clave principales
        key_points = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130]

        height, width = gray_image.shape

        # Crear una nueva figura
        plt.clf()
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(gray_image, cmap='gray')

        # Dibujar puntos faciales
        for point_idx in key_points:
            landmark = results.multi_face_landmarks[0].landmark[point_idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            plt.plot(x, y, 'rx')

        # Guardar el resultado en memoria
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Convertir a base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

    except Exception as e:
        print(f"Error en analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')


def augment_data(keyfacial_df):
    try:
        # Crear una copia del dataframe
        keyfacial_df_copy = copy.copy(keyfacial_df)

        # Voltear horizontalmente
        keyfacial_df_copy['Image'] = keyfacial_df['Image'].apply(lambda x: np.flip(x, axis=1))
        columns = keyfacial_df.columns[:-1]

        for i in range(len(columns)):
            if i % 2 == 0:
                keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x))

        # Aumentar el brillo aleatoriamente
        bright_df = copy.copy(keyfacial_df)
        bright_df['Image'] = bright_df['Image'].apply(lambda x: np.clip(random.uniform(1.5, 2) * x, 0.0, 255.0))

        # Combinar datos
        augmented_df = np.concatenate((keyfacial_df, keyfacial_df_copy, bright_df))
        return augmented_df

    except Exception as e:
        print(f"Error en augment_data: {str(e)}")
        raise


@app.route('/')
def home():
    images = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    return render_template('index.html', images=images)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Subir un archivo nuevo
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No se seleccionó un archivo'}), 400

            if not allowed_file(file.filename):
                return jsonify({'error': 'Tipo de archivo no permitido'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

        # Analizar la imagen
        result_image = analyze_face(filepath)

        return jsonify({'success': True, 'image': result_image})

    except Exception as e:
        print(f"Error en /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
