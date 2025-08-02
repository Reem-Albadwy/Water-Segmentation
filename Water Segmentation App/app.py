from flask import Flask, render_template, request
import os
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
import tifffile as tiff
import cv2
from fpdf import FPDF
import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PREDICTED_FOLDER = 'static/predicted'
PROCESSED_FOLDER = 'static/processed'
OVERLAY_FOLDER = 'static/overlay'
PDF_FOLDER = 'static/pdf'
MODEL_PATH = 'Segmentation_Feature_Extraction.h5'

for folder in [UPLOAD_FOLDER, PREDICTED_FOLDER, PROCESSED_FOLDER, OVERLAY_FOLDER, PDF_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_URL = "https://huggingface.co/Reem1/Segmentation_Feature_Extraction.h5/resolve/main/Segmentation_Feature_Extraction.h5"

class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

# Load model from huggingface
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

def calculate_coverage(mask):
    total_pixels = mask.shape[0] * mask.shape[1]
    water_pixels = np.sum(mask > 127)
    return round((water_pixels / total_pixels) * 100, 2)

def generate_overlay(rgb_img, mask):
    mask_colored = np.zeros_like(rgb_img)
    mask_colored[:, :, 0] = mask 

    overlay = cv2.addWeighted(rgb_img, 0.75, mask_colored, 0.25, 0)
    return overlay

def generate_pdf(original_path, mask_path, overlay_path, percentage, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Water Segmentation Report ", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Water Coverage: {percentage}%", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.image(original_path, x=10, y=50, w=60)
    pdf.image(mask_path, x=75, y=50, w=60)
    pdf.image(overlay_path, x=140, y=50, w=60)
    pdf.output(output_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_result = None
    rgb_result = None
    overlay_result = None
    coverage_percent = None
    pdf_report_url = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename.rsplit('.', 1)[0]
            tif_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(tif_path)

            img = tiff.imread(tif_path)
            water_index = calculate_water_index(img)
            full_img = np.concatenate([img, water_index], axis=-1)

            # Normalize
            min_val = full_img.min(axis=(0, 1), keepdims=True)
            max_val = full_img.max(axis=(0, 1), keepdims=True)
            scaled_img = (full_img - min_val) / (max_val - min_val + 1e-8)
            input_img = np.expand_dims(scaled_img, axis=0)

            # Predict
            prediction = model.predict(input_img)[0]
            prediction_binary = (prediction > 0.5).astype(np.uint8) * 255
            prediction_resized = cv2.resize(prediction_binary, (128, 128))

            # Save mask
            pred_filename = f"pred_{filename}.png"
            pred_path = os.path.join(PREDICTED_FOLDER, pred_filename)
            pred_result = os.path.join('static/predicted', pred_filename)
            cv2.imwrite(pred_path, prediction_resized)

            # Calculate %
            coverage_percent = calculate_coverage(prediction_resized)

            # Create RGB image
            rgb = img[:, :, [3, 2, 1]]
            rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
            rgb = (rgb * 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb_filename = f"rgb_{filename}.png"
            rgb_path = os.path.join(PROCESSED_FOLDER, rgb_filename)
            rgb_result = os.path.join('static/processed', rgb_filename)
            cv2.imwrite(rgb_path, rgb_bgr)

            # Overlay
            overlay_img = generate_overlay(rgb_bgr, prediction_resized)
            overlay_filename = f"overlay_{filename}.png"
            overlay_path = os.path.join(OVERLAY_FOLDER, overlay_filename)
            overlay_result = os.path.join('static/overlay', overlay_filename)
            cv2.imwrite(overlay_path, overlay_img)

            # PDF report
            pdf_filename = f"report_{filename}.pdf"
            pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
            pdf_report_url = os.path.join('static/pdf', pdf_filename)
            generate_pdf(rgb_path, pred_path, overlay_path, coverage_percent, pdf_path)

            return render_template(
                "index.html",
                pred_result=pred_result,
                rgb_result=rgb_result,
                overlay_result=overlay_result,
                coverage_percent=coverage_percent,
                pdf_report_url=pdf_report_url
            )

    return render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))



