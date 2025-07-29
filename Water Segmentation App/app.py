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

def calculate_iou(true_mask, pred_mask):
    true_mask = true_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()

    if union == 0:
        return 100.0 if intersection == 0 else 0.0

    iou = intersection / union
    return round(iou * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    pred_result = None
    rgb_result = None
    true_mask = None
    iou_score = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = tiff.imread(filepath)  # (128,128,12)
            water_index = calculate_water_index(img)
            full_img = np.concatenate([img, water_index], axis=-1)  # (128,128,13)
            input_img = np.expand_dims(full_img, axis=0)  # (1,128,128,13)

            # Prediction
            prediction = model.predict(input_img)[0]
            prediction_binary = (prediction > 0.5).astype(np.uint8) * 255
            prediction_resized = cv2.resize(prediction_binary, (128, 128))

            pred_filename = 'pred_' + filename.rsplit('.', 1)[0] + '.png'
            pred_path = os.path.join(PREDICTED_FOLDER, pred_filename)
            cv2.imwrite(pred_path, prediction_resized)

            # Extract RGB for display
            rgb = img[:, :, [3, 2, 1]]
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb_path = os.path.join(PROCESSED_FOLDER, 'rgb_image.png')
            cv2.imwrite(rgb_path, rgb_bgr)

            # Ground truth mask from band 12
            gt_mask = img[:, :, 11]
            gt_mask = (gt_mask > 0).astype(np.uint8) * 255
            gt_path = os.path.join(PROCESSED_FOLDER, 'true_mask.png')
            cv2.imwrite(gt_path, gt_mask)

            # IoU
            iou_score = calculate_iou(gt_mask, prediction_resized)

            return render_template(
                "index.html",
                result=filepath,
                pred_result=pred_path,
                rgb_result=rgb_path,
                true_mask=gt_path,
                iou_score=iou_score
            )

    return render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
