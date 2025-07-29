from flask import Flask, render_template, request
import os
import numpy as np
import requests
from tensorflow.keras.models import load_model
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
if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
        
model = load_model(MODEL_PATH, compile=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    pred_result = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = load_img(filepath, target_size=(128, 128))
            image = img_to_array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = model.predict(image)[0]
            prediction = (prediction > 0.5).astype(np.uint8) * 255
            prediction = cv2.resize(prediction, (128, 128))

            pred_path = os.path.join(app.config['PREDICTED_FOLDER'], 'pred_' + file.filename)
            cv2.imwrite(pred_path, prediction)

            result = filepath
            pred_result = pred_path

    return render_template('index.html', result=result, pred_result=pred_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))



