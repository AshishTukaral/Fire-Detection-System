from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8n_custom_fire_2/weights/best.pt")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Get the width and height of the display window
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#print(screen_width,screen_height)

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

while cap.isOpened():
    ret, frame = cap.read()

    # Perform object detection on the frame
    results = model.predict(source=frame, line_width=1, show_conf=False)

    # Extract detection results
    boxes = results[0].boxes.xywh.cpu()  # xywh bbox list
    clss = results[0].boxes.cls.cpu().tolist()  # classes Id list
    names = results[0].names  # classes names list
    confs = results[0].boxes.conf.float().cpu().tolist()  # probabilities of classes

    for box, cls, conf in zip(boxes, clss, confs):
        x, y, w, h = box
        percentage = int(conf * 100)  # Calculate detection percentage

        # Display the bounding box and label on the frame
        cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255), 2)

        # Display the danger message at the bottom center of the screen
        cv2.putText(frame, 'Danger', (225, screen_height - 38), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4,
                    cv2.LINE_AA)

        cv2.putText(frame, 'Fire Detected', (241, screen_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                    2, cv2.LINE_AA)

    # Display the project name at the top center of the screen
    cv2.putText(frame, 'Project: Fire Detection & Monitoring', (26, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255),3, cv2.LINE_AA)

    # Display the processed frame
    cv2.imshow('Fire Detection', frame)

    # Check for the 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
