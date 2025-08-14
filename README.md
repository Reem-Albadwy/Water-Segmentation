# 💧 Water Segmentation using Deep Learning

This project focuses on image segmentation to detect **water bodies** in satellite images using deep learning. It includes a model built from scratch and another leveraging **EfficientNetB0** as a pretrained encoder. The final model is deployed using **Flask** through a simple web interface.

##  Key Highlights

- Built a U-Net deep learning model from scratch.
- Applied **transfer learning** by using EfficientNetB0 as the encoder for U-Net.
- Engineered a new **Water Index feature** to improve segmentation quality.
- Used **MinMaxScaler** for input normalization.
- Developed a **Flask** web app for user-friendly deployment and image uploading.

##  Model Details

- **Model 1:** U-Net from scratch with custom convolutional layers.
- **Model 2:** U-Net with pretrained **EfficientNetB0** encoder (ImageNet weights).
- **Loss Function:** Jaccard Loss  
- **Optimizer:** Adam  
- **Metric:** IoU Score  
- **Input Shape:** 128×128×12  
- **Normalization:** MinMaxScaler  
- **Additional Feature:** Water Index added to input channels

##  Test Results

- **Test Loss:** `0.31085`
- **Test IoU Score:** `0.82624`

##  Deployment

- The model is deployed using a **Flask** app.
- Users can upload satellite images and receive predicted **water masks**.

### 🔗 Links

- **Model Weights (Hugging Face):**  
  [Segmentation_Feature_Extraction.h5](https://huggingface.co/Reem1/Segmentation_Feature_Extraction.h5/resolve/main/Segmentation_Feature_Extraction.h5)

- **Live Web App:**  
  [Water Segmentation App](https://water-segmentation-production-3f21.up.railway.app/)
