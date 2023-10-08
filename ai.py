from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Conv2D
from tensorflow.keras.applications import MobileNetV2

# Define the number of classes in your classification task
num_classes = 3  # Replace with your actual number of classes

# Define the base model (e.g., MobileNetV2)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Reshape the input data to match the base model's input shape
input_tensor = Input(shape=(224, 224, 3))  # Replace with your input shape
x = base_model(input_tensor)

# Add additional convolutional layers if needed
x = Conv2D(256, (3, 3), activation='relu')(x)
x = GlobalAveragePooling2D()(x)

# Define the classification head (replace num_classes with your number of classes)
class_output = Dense(num_classes, activation='softmax', name='class_output')(x)

# Define the regression head for bounding box coordinates (4 values: x, y, width, height)
bbox_output = Dense(4, activation='linear', name='bbox_output')(x)

# Create the final model with both outputs
model = Model(inputs=input_tensor, outputs=[class_output, bbox_output])

# Print the expected input shape of the base model
print("Expected input shape of base model:", base_model.input_shape)
model.compile(optimizer='adam',
              loss={'class_output': 'categorical_crossentropy', 'bbox_output': 'mean_squared_error'},
              loss_weights={'class_output': 1.0, 'bbox_output': 1.0})
model.save("ss.h5")