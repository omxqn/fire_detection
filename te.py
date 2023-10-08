from keras.models import load_model
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Specify the path to the input file (image or MP4 video)
input_path = "test.mp4"  # Replace with the path to your input file (image or video)

# Check the file extension to determine whether it's an image or video
if input_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    # Read an image
    image = cv2.imread(input_path)

    if image is None:
        print("Failed to load the image. Please check the file path.")
    else:
        # Resize the raw image into (224-height, 224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the model's input shape.
        image_for_prediction = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image_for_prediction = (image_for_prediction / 127.5) - 1

        # Predict the model
        prediction = model.predict(image_for_prediction)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:])
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        # Draw a bounding box around the detected fire (assuming class_name is "fire")
        if class_name[2:][:4].lower() == 'fire':
            print("drawing")
            # Define the color and thickness of the bounding box and text
            color = (0, 0, 255)  # Red color in BGR
            thickness = 2
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Define the coordinates of the bounding box
            x, y, w, h = 50, 50, 100, 100  # Adjust these values based on your detection results

            # Draw the bounding box on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

            # Draw the "Fire" text above the bounding box
            text = "Fire"
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (x + (w - text_size[0]) // 2)-20
            text_y = y - 10  # Adjust this value to control the vertical position of the text
            cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

            text = f"Accuracy: {str(np.round(confidence_score * 100))[:-2]}%"
            text_size = cv2.getTextSize(text, font, font_scale-0.2, thickness)[0]
            text_x = (x + (w - text_size[0]) // 2) + 40
            text_y = y - 10  # Adjust this value to control the vertical position of the text
            cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

        if class_name[2:][:7].lower() == 'smoking':
            print("drawing")
            # Define the color and thickness of the rectangle
            color = (128, 128, 128)  # Gray color in BGR
            thickness = 2

            # Define the coordinates of the rectangle
            x, y, w, h = 50, 50, 100, 100  # Adjust these values based on your detection results

            # Draw the gray rectangle on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

            # Scale the image to its original size
            original_size_image = cv2.imread(input_path)

            # Show the original size image with the gray rectangle
            cv2.imshow("Original Size Image with Rectangle", original_size_image)

        # Show the image with the bounding box
        cv2.imshow("Image", image)

        # Wait for a key press and then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
elif input_path.endswith('.mp4'):
    # Read an MP4 video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Failed to open the video file. Please check the file path.")
    else:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Resize the frame into (224-height, 224-width) pixels
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

            # Make the frame a numpy array and reshape it to the model's input shape.
            frame_for_prediction = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalize the frame array
            frame_for_prediction = (frame_for_prediction / 127.5) - 1

            # Predict the model
            prediction = model.predict(frame_for_prediction)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Print prediction and confidence score
            print("Class:", class_name[2:])
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

            # Draw a bounding box around the detected fire (assuming class_name is "fire")
            if class_name[2:][:4].lower() == 'fire':
                print("drawing")
                # Define the color and thickness of the bounding box and text
                color = (0, 0, 255)  # Red color in BGR
                thickness = 2
                font_scale = 0.5
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Define the coordinates of the bounding box
                x, y, w, h = 50, 50, 100, 100  # Adjust these values based on your detection results

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                # Draw the "Fire" text above the bounding box
                text = "Fire"
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (x + (w - text_size[0]) // 2)-20
                text_y = y - 10  # Adjust this value to control the vertical position of the text
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

                text = f"Accuracy: {str(np.round(confidence_score * 100))[:-2]}%"
                text_size = cv2.getTextSize(text, font, font_scale-0.2, thickness)[0]
                text_x = (x + (w - text_size[0]) // 2) + 40
                text_y = y - 10  # Adjust this value to control the vertical position of the text
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

            cv2.imshow("Video Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
