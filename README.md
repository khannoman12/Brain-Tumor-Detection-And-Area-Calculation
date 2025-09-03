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

# 📂 Project Structure
├── src/
│   ├── app.py        # Streamlit app
│   ├── train.py      # Training code
├── notebooks/        # Experiment notebooks
├── models/           # YOLO weights (best.pt, best.onnx)
├── requirements.txt  # Dependencies
└── README.md         # Documentation

# 🛠️ Tech Stack
Python (3.10)

PyTorch

Ultralytics YOLOv11s

OpenCV

NumPy / Pandas

Streamlit

