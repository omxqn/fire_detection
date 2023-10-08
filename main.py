import cv2
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def run_baseline(model_name):
    classes = ['fire', 'non_fire']
    detected = False

    img_size = 224
    # Loading the trained fire classification model
    model = load_model(model_name)

    # OpenCV VideoCapture object
    cap = cv2.imread("video_frames/frame_0000.jpg")

    # Initialize frame count and variables for calculating accuracy
    frame = 0
    first_frame = np.nan
    time_list = []


    while True:
        image = cap

        cv2.imshow("sd", image)
        cv2.waitKey(1)

        # Preprocess the image according to the model
        image = cv2.resize(image, (img_size, img_size))
        if model_name.lower() == 'firenet':
            image = image.astype("float") / 255.0
        elif model_name.lower() == 'mobilenet':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        if model_name.lower() == 'mobilenet':
            image = keras.applications.mobilenet.preprocess_input(image)

        tic = time.time()
        # Inference
        softmax_output = model.predict(image)
        toc = time.time()
        det_time = toc - tic
        print("Time taken =", det_time)
        time_list.append(det_time)

        # Output prediction from softmax
        idx = np.argmax(softmax_output)
        prediction = classes[idx]
        print(f"Prediction: {prediction}")
        if prediction.lower() == 'fire':
            if not detected:
                first_frame = frame
            detected = True

        break

    output = pd.DataFrame({
        'Source': ["Sdfsdf"],
        'detected': [detected],
        'first_frame': [first_frame],
        'time_avg': [np.mean(time_list)]
    })

    return output


if __name__ == "__main__":

    # Run inference
    result = run_baseline(model_name="keras_model.h5")
    print("Results")
    print(result)
