import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(dataset_path):
    images = []
    labels = []

    for label in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, label)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Black and white reading
                if image is not None:
                    image = cv2.resize(image, (48, 48))  # Resize image to 48x48
                    images.append(image)
                    labels.append(label)

    return np.array(images), np.array(labels)

def create_emotion_model(num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))  # Single channel for black and white
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # For emotion categories

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Determining the dataset path according to the directory where the code is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'dataset')  # The folder containing the emotion data
    if not os.path.exists(dataset_path):
        print(f"'dataset' folder not found: {dataset_path}")
        return

    images, labels = load_data(dataset_path)

    # Converting labels to numeric values
    label_to_index = {label: index for index, label in enumerate(sorted(set(labels)))}
    y = np.array([label_to_index[label] for label in labels])
    y = to_categorical(y)  # One-hot encoding

    # Converting data to numpy arrays and normalizing
    X = np.array(images, dtype='float32') / 255.0  # Normalization to 0-1 range
    X = np.expand_dims(X, axis=-1)  # Add channel size for black and white

    # Separating training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Create the model
    model = create_emotion_model(len(label_to_index))

    # Early stop
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=150, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Save the model
    model.save(os.path.join(base_dir, 'emotion_recognition_model.h5'))

    # Save tags
    with open(os.path.join(base_dir, 'labels.txt'), 'w') as f:
        for label, index in label_to_index.items():
            f.write(f"{index}:{label}\n")

if __name__ == "__main__":
    main()
