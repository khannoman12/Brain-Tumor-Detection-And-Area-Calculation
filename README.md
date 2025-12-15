# Brain-Tumor-Detection-And-Area-Calculation

deep learning project for brain tumor detection, classification, segmentation, and tumor size estimation using YOLOv11s and Streamlit.

# Overview
This project detects and segments brain tumors from MRI images using YOLOv11s and calculates tumor size as a percentage of brain area.
It was developed as my Final Year Project (BS Data Science, UET Peshawar) in collaboration with Hayatabad Medical Complex (HMC), under the supervision of Dr. Junaid Alam (Neurosurgeon) and support from CISNR, UET Peshawar

# 🚀 Features

Detects 3 tumor types: Glioma, Meningioma, Pituitary

Tumor segmentation with mask overlay

Tumor size estimation (% of brain area) using OpenCV contours

Real-time inference via Streamlit web app

Model exportable to ONNX for deployment

# 🛠️ Tech Stack
Python (3.10)

PyTorch

Ultralytics YOLOv11s

OpenCV

NumPy / Pandas

Streamlit

#📊 Dataset

3,068 MRI brain images collected from HMC, Peshawar

Annotated with Roboflow under expert supervision

Augmentation expanded dataset to 7,230 images

Augmentations: rotation, shear, brightness, exposure, noise

# 📈 Training Details

Platform: Kaggle Free GPU (12-hour limit)

Epochs: 100 (~10h 45m)

Optimizer: SGD

Initial segmentation loss: ~65% → reduced to ~20%

Final Accuracy: 85% Precision, 84% Recall, mAP50 = 0.88 

## 📷 Interface
https://brain-tumor-detection-and-area-calculation-pvhkcf9erqfy8hasry3.streamlit.app/





