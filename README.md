# Emotion Recognition from Facial Expressions using TensorFlow & OpenCV

## Project Overview

This project implements a real-time facial emotion recognition system using deep learning.  
It utilizes a convolutional neural network (CNN) trained on grayscale facial images to classify emotions such as happy, sad, angry, surprised, and neutral.  
The system captures live video from a webcam, detects faces using OpenCV's Haar cascades, and predicts the emotional state of detected faces with confidence scores displayed on the video feed.

---

## Features

- Real-time emotion detection with webcam input  
- Face detection using OpenCV Haar cascades  
- CNN model trained with TensorFlow/Keras on grayscale images  
- Data augmentation and early stopping to improve training robustness  
- Easy to extend with custom datasets

---

## Requirements

- Python 3.8 or higher  
- Packages listed in `requirements.txt`:  
  - tensorflow==2.19.0  
  - opencv-python==4.11.0.86  
  - scikit-learn==1.6.1  
  - h5py==3.13.0  
  - numpy  

---

## Installation

1. Clone this repository:
   
```bash git clone https://github.com/CayMan4444/Face-Emotion-Recognition/
