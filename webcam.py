import cv2
import torch
import os

# Load models
try:
    model_1 = torch.hub.load('E:/Object_detection/yolov5', 'custom', path='E:/Object_detection/yolov5/runs/train/exp26/weights/best.pt', source='local')
    print("Model 1 loaded successfully!")
except Exception as e:
    print(f"Error loading Model 1: {e}")

try:
    model_2 = torch.hub.load('E:/Object_detection/yolov5', 'custom', path='E:/Object_detection/yolov5/runs/train/exp28/weights/best.pt', source='local')
    print("Model 2 loaded successfully!")
except Exception as e:
    print(f"Error loading Model 2: {e}")

# Choose input type
input_type = input("Choose input type ('webcam' or 'file'): ").lower()

if input_type == 'w':
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Open the default webcam
    if not cap.isOpened():
        print("Error: Could not access the webcam")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam")
            break

        # Run inference with only Model 1
        print("Running inference with Model 1...")
        results_1 = model_1(frame)

        # Render the results (overlay bounding boxes)
        results_1.render()

        # Show the results
        cv2.imshow('YOLOv5 Object Detection - Model 1', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

elif input_type == 'f':
    # Handle file input
    file_path = input("Enter the file path of the image/video: ")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} does not exist.")
        exit()

    # Read the file (image or video)
    if file_path.endswith(('mp4', 'avi', 'mov', 'mkv')):
        # Process video file
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Could not open the video file")
            exit()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from video")
                break

            # Run inference with only Model 1
            print("Running inference with Model 1...")
            results_1 = model_1(frame)

            # Render the results (overlay bounding boxes)
            results_1.render()

            # Show the results
            cv2.imshow('YOLOv5 Object Detection - Model 1', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif file_path.endswith(('jpeg', 'jpg', 'png', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv', 'avi')):
        # Process image file
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Could not read the image from {file_path}")
            exit()

        # Run inference with only Model 1
        print("Running inference with Model 1...")
        results_1 = model_1(image)

        # Render the results (overlay bounding boxes)
        results_1.render()

        # Show the results
        cv2.imshow('YOLOv5 Object Detection - Model 1', image)

        # Wait for a key press to close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Unsupported file format. Only image and video formats like jpeg, jpg, png, mp4, avi, mov, and mkv are supported.")
else:
    print("Invalid input type. Exiting.")
