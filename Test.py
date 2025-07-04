import cv2
from ultralytics import YOLO

def predict_real_time(model_path, tomato_classes):
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Open the default camera (ID 0). Change ID for other cameras
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Camera accessed successfully. Press 'q' to quit.")

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to fetch frame.")
            break

        # Make predictions on the frame
        results = model.predict(source=frame, save=False, show=False, conf=0.25)

        # Annotate the frame
        annotated_frame = frame.copy()
        for result in results[0].boxes.data:
            # Extract box coordinates, confidence, and class ID
            x1, y1, x2, y2, conf, cls_id = result.tolist()
            cls_id = int(cls_id)

            # Check if the class ID is one of the tomato-related classes
            if cls_id < len(tomato_classes) and tomato_classes[cls_id] in tomato_classes:
                label = tomato_classes[cls_id]
                color = (0, 255, 0)  # Green for tomato-related classes
            else:
                label = "Not a Tomato"
                color = (0, 0, 255)  # Red for non-tomato objects

            # Draw the bounding box and label
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated_frame, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame with annotations
        cv2.imshow("Real-Time Tomato Ripeness Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Define your tomato-related classes
    tomato_classes = [
        "b_fully_ripened", "b_half_ripened", "b_green",
        "l_fully_ripened", "l_half_ripened", "l_green"
    ]

    # Provide the path to the best-trained model
    model_path = r"C:\Users\ACER\runs\detect\train19\weights\best.pt"
    predict_real_time(model_path, tomato_classes)
