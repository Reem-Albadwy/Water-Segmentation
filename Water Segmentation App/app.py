from flask import Flask, render_template, request
import os
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.activations import swish
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import tifffile as tiff

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PREDICTED_FOLDER = 'static/predicted'
PROCESSED_FOLDER = 'static/processed'
MODEL_PATH = 'Segmentation_Feature_Extraction.h5'

for folder in [UPLOAD_FOLDER, PREDICTED_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTED_FOLDER'] = PREDICTED_FOLDER

MODEL_URL = "https://huggingface.co/Reem1/Segmentation_Feature_Extraction.h5/resolve/main/Segmentation_Feature_Extraction.h5"

class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)

model = load_model(MODEL_PATH, compile=False, custom_objects={'swish': tf.nn.swish, 'FixedDropout': FixedDropout})

def calculate_water_index(img):
    R = img[:, :, 3].astype(np.float32)
    G = img[:, :, 2].astype(np.float32)
    B = img[:, :, 1].astype(np.float32)
    water_index = (G - R) / (G + R + 1e-5)
    return np.expand_dims(water_index, axis=-1)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    pred_result = None
    rgb_result = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = tiff.imread(filepath)  # (128,128,12)
            water_index = calculate_water_index(img)
            full_img = np.concatenate([img, water_index], axis=-1)  # (128,128,13)
            # Normalize like notebook (per image)
            min_val = full_img.min(axis=(0, 1), keepdims=True)
            max_val = full_img.max(axis=(0, 1), keepdims=True)
            scaled_img = (full_img - min_val) / (max_val - min_val + 1e-8)
            input_img = np.expand_dims(scaled_img, axis=0)

            # Prediction
            prediction = model.predict(input_img)[0]
            prediction_binary = (prediction > 0.5).astype(np.uint8) * 255
            prediction_resized = cv2.resize(prediction_binary, (128, 128))

            pred_filename = 'pred_' + filename.rsplit('.', 1)[0] + '.png'
            pred_result = os.path.join('static/predicted', pred_filename)
            cv2.imwrite(pred_path, prediction_resized)

            # Extract RGB for display
            rgb = img[:, :, [3, 2, 1]]
            rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))  # min-max scale
            rgb = (rgb * 255).astype(np.uint8)

            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb_filename = 'rgb_' + filename.rsplit('.', 1)[0] + '.png'
            rgb_path = os.path.join(PROCESSED_FOLDER, rgb_filename)
            rgb_result = os.path.join('static/processed', rgb_filename)
            cv2.imwrite(rgb_path, rgb_bgr)

            return render_template(
                "index.html",
                result=filepath,
                pred_result=pred_path,
                rgb_result=rgb_path,
            )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
