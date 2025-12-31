# Real-Time-Face-Pixelation-for-Privacy-Protection

This project implements a real-time face anonymization system using
deep learning-based face detection and pixelation techniques.
The goal is to protect user identity in images and live video streams,
with a focus on privacy-preserving computer vision applications.

---

## 🔍 Project Overview

Traditional face detection methods often fail when faces are rotated or partially occluded.
To address this limitation, this project uses a deep learning-based face detector
(OpenCV DNN with an SSD architecture) combined with pixelation-based anonymization.

The system detects faces in real time and applies pixelation to prevent identity recognition,
while maintaining stable performance in live video streams.

---

## ✨ Features

- Deep learning-based face detection (CNN / SSD)
- Robust detection under pose variations
- Real-time face pixelation for identity protection
- Adjustable pixelation strength
- Webcam-based live processing

---

## 🧠 Technologies Used

- **Python**
- **OpenCV**
- **OpenCV DNN module**
- **Pre-trained SSD face detector (Caffe format)**

---

## 📁 Project Structure

face_pixelation_project/
<br/>&nbsp;&nbsp;&nbsp;face_pixelation_dnn.py
<br/>&nbsp;&nbsp;&nbsp;deploy.prototxt
<br/>&nbsp;&nbsp;&nbsp;res10_300x300_ssd_iter_140000.caffemodel
<br/>&nbsp;&nbsp;&nbsp;requirements.txt
<br/>&nbsp;&nbsp;&nbsp;README.md


---

## 🚀 How to Run

### 1. Create and activate a Conda environment
```bash
conda create -n face_pixelation_env 
conda activate face_pixelation_env
```

### 2. Install dependencies

```bash
pip install opencv-python
```

### 3. Run the application
```bash
python face_pixelation_dnn.py
```

Press q to exit the program
