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
   
```bash
git clone https://github.com/CayMan4444/Face-Emotion-Recognition/
```

2. (Optional) Create and activate a virtual environment:
Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
Prepare your dataset folder as follows:
```bash
dataset/
├── angry/
├── happy/
├── sad/
├── surprise/
└── neutral/
```
Each subfolder should contain grayscale images of faces expressing the respective emotion.

Run the training script:
```bash
python train_model.py
```
The model will train with data augmentation and early stopping.
Once completed, it saves:
 * `emotion_recognition_model.h5` — the trained CNN model
 * `labels.txt` — mapping of label indices to emotion names

## Running Real-Time Emotion Detection

Run the detection script to start webcam emotion recognition:
```bash
python emotion_detection.py
```
A window will open showing webcam video with detected faces and their predicted emotions.
Press `q` to quit the program.

## Project Structure

* `train_model.py`: Script to load data, train the CNN model, and save model & labels
* `emotion_detection.py`: Script to run live emotion recognition via webcam
* `dataset/`: Folder containing emotion-labeled training images
* `requirements.txt`: Python dependencies
* `labels.txt`: Label-to-index mapping saved after training
* `emotion_recognition_model.h5`: Saved trained CNN model

## Troubleshooting

* Ensure your webcam is connected and accessible by OpenCV.
* Verify that `emotion_recognition_model.h5` and `labels.txt` are in the same directory as `emotion_detection.py` or update paths accordingly.
* If package installation fails, check your Python version and pip installation.

## Contribution

Contributions and improvements are welcome! Please fork the repository, create a feature branch, and submit a pull request.
Maintain clear commit messages and update documentation as needed.

## License

This project is licensed under the MIT License.

## Acknowledgements

* OpenCV for face detection utilities
* TensorFlow/Keras for deep learning framework
* scikit-learn for data preprocessing
* Inspired by common emotion recognition models and datasets
