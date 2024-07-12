import cv2
import numpy as np
import os

# Define the paths to the model files
prototxt = "C:/Users/HARSH/OneDrive/Desktop/Deepthi/MobileNetSSD_deploy.prototxt"
model = "C:/Users/HARSH/OneDrive/Desktop/Deepthi/MobileNetSSD_deploy.caffemodel"

# Check if the model files exist
if not os.path.isfile(prototxt):
    print(f"Error: The file {prototxt} does not exist.")
    exit(1)
if not os.path.isfile(model):
    print(f"Error: The file {model} does not exist.")
    exit(1)

# Load the pre-trained model
try:
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
except cv2.error as e:
    print(f"Error loading model: {e}")
    exit(1)

# Open the video file
video_path = "C:/Users/HARSH/OneDrive/Desktop/Deepthi/video2.mp4"
if not os.path.isfile(video_path):
    print(f"Error: The file {video_path} does not exist.")
    exit(1)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video {video_path}")
    exit(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Prepare the frame for detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Forward pass to get the detections
    try:
        detections = net.forward()
    except cv2.error as e:
        print(f"Error during forward pass: {e}")
        break

    # Initialize counter for the number of people
    num_people = 0

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])

            # Check if the detected object is a person (class label 15 in MobileNet SSD)
            if idx == 15:
                num_people += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box and label on the frame
                label = f"Person: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the number of people detected on the frame
    cv2.putText(frame, f"Number of people detected: {num_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Output", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
