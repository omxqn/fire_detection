import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Data directories
fire_images_dir = "Data/Train_Data/Fire"
non_fire_images_dir = "Data/Train_Data/Non_Fire"
video_frames_dir = "video_frames"

# Function to load and preprocess images
def load_and_preprocess_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, filename))
        if image is not None:
            image = cv2.resize(image, IMAGE_SIZE)  # Resize to a common size
            images.append(image)
    return images

# Load and preprocess image data
fire_images = load_and_preprocess_images(fire_images_dir)
non_fire_images = load_and_preprocess_images(non_fire_images_dir)

# Labels: 1 for fire, 0 for non-fire
labels = [1] * len(fire_images) + [0] * len(non_fire_images)

# Combine image data
images = np.array(fire_images + non_fire_images, dtype=np.float32)
images /= 255.0  # Normalize pixel values

# Create dummy target labels for video frames (assuming all frames are fire-related)
video_labels = [1] * len(os.listdir(video_frames_dir))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create the model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the combined data
model.fit(
    X_train,  # Use X_train as input data
    np.array(y_train),  # Convert y_train to a numpy array
    batch_size=BATCH_SIZE,
    validation_data=(X_val, np.array(y_val)),  # Convert y_val to a numpy array
    epochs=EPOCHS,
)

# Save the trained model
model.save("fire_detection_model.h5")
