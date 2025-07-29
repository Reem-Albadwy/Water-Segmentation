from flask import Flask, render_template, request
import os
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.activations import swish
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PREDICTED_FOLDER = 'static/predicted'
MODEL_PATH = 'Segmentation_Feature_Extraction.h5'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTED_FOLDER'] = PREDICTED_FOLDER

MODEL_URL="https://huggingface.co/Reem1/Segmentation_Feature_Extraction.h5/resolve/main/Segmentation_Feature_Extraction.h5"

class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        
if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
        
model = load_model(MODEL_PATH, compile=False, custom_objects={'swish': tf.nn.swish, 'FixedDropout': FixedDropout})

@app.route('/', methods=['GET', 'POST'])
def calculate_water_index(img):
    R = img[:, :, 3].astype(np.float32)
    G = img[:, :, 2].astype(np.float32)
    B = img[:, :, 1].astype(np.float32)
    water_index = (G - R) / (G + R + 1e-5)
    return np.expand_dims(water_index, axis=-1)

def index():
    result = None
    pred_result = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = np.array(Image.open(filepath))
            water_index = calculate_water_index(img)
            img = np.concatenate([img, water_index], axis=-1)  # (128,128,13)
            img = np.expand_dims(img, axis=0)  # (1,128,128,13)

            prediction = model.predict(img)[0]
            prediction = (prediction > 0.5).astype(np.uint8) * 255
            prediction = cv2.resize(prediction, (128, 128))

            pred_filename = 'pred_' + filename.rsplit('.', 1)[0] + '.png'
            pred_path = os.path.join(app.config['PREDICTED_FOLDER'], pred_filename)
            cv2.imwrite(pred_path, prediction)

            result = filepath
            pred_result = pred_path

    return render_template('index.html', result=result, pred_result=pred_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))



