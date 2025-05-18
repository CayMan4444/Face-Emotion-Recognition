import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import h5py

def load_labels(label_file):
    """Loads the tag file."""
    label_to_index = {}
    with open(label_file, 'r') as f:
        for line in f:
            index, label = line.strip().split(':')
            label_to_index[int(index)] = label
    return label_to_index

def preprocess_image(image):
    """Makes the image fit the model."""
    image = cv2.resize(image, (48, 48))  # Size accepted by the model
    image = image.astype('float32') / 255.0  # Normalization
    image = np.expand_dims(image, axis=-1)  # Channel size for black and white
    image = np.expand_dims(image, axis=0)  # Adapting the model to the input format (batch size)
    return image

def predict_emotion(model, labels, image):
    """It makes emotion prediction by processing the image."""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_label = np.argmax(predictions)
    emotion = labels[predicted_label]
    confidence = predictions[0][predicted_label] * 100
    return emotion, confidence

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model/emotion_recognition_model.h5')
    label_file = os.path.join(base_dir, 'model/labels.txt')

    # Checking if the file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Check HDF5 compatibility
    try:
        with h5py.File(model_path, 'r') as f:
            print("HDF5 format checked.")
    except Exception as e:
        print(f"The model file is in an invalid format: {e}")
        return

    # Load the model and labels
    model = load_model(model_path)
    print("Model loaded successfully.")
    labels = load_labels(label_file)
    print("Tags loaded successfully.")

    # Real-time emotion recognition with camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera!")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            try:
                emotion, confidence = predict_emotion(model, labels, face)
                label = f"{emotion} ({confidence:.1f}%)"

                # Kare içine alınan yüz
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            except Exception as e:
                print(f"Error while processing face: {e}")

        cv2.imshow('Emotion Recognition', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
