import cv2
import os
# Exctact the frames from the mp4 to training
# Path to the input video
video_path = "test.mp4"

# Directory to store the extracted video frames
video_frames_dir = "video_frames"

# Create the video_frames directory if it doesn't exist
if not os.path.exists(video_frames_dir):
    os.makedirs(video_frames_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize frame count
frame_count = 0

# Read and save frames from the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save each frame as an image in the video_frames directory
    frame_filename = os.path.join(video_frames_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Release the video capture object
cap.release()
