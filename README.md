# Fire Detection System

## Introduction:
This project aims to develop a fire detection system using the YOLOv8 model for real-time detection and tracking of fires in video streams. The system draws bounding boxes around detected fires and displays a "Danger: Fire Detected" message on the video feed.

## Fire Dataset Link:
The dataset used for training the YOLOv8 model can be accessed at [Fire Dataset](https://mega.nz/file/MgVhQSoS#kOcuJFezOwU_9F46GZ1KJnX1STNny-tlD5oaJ9Hv0gY).

## Instructions to run the script

1. Install Required Libraries:
   - Ensure that you have Python installed on your system.
   - Navigate to the project directory in your terminal.
   - Install the required libraries using the provided `requirements.txt` file by running the following command:
     ```
     pip install -r requirements.txt
     ```

2. Train the Model:
   - Before making predictions, you need to train the YOLOv8 model using the provided dataset and configuration.
   - Execute the model training script using the following command:
     ```
     python train_model.py
     ```
   - Adjust training parameters such as epochs, batch size, and model name as needed in the script.

3. Make Predictions:
   - After training the model, you can make predictions on video streams or images to detect fires.
   - Run the prediction script using the following command:
     ```
     python fire_detection.py
     ```
   - Ensure that the trained model weights (e.g., `best.pt`) are correctly placed in the designated directory.
   - The script will capture video frames from the webcam, perform fire detection using the trained model, and display the results in real-time.

## Training the YOLOv8 Model:
1. Load the YOLOv8 model using the following code:
   ```
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   ```

2. Start training the model using the `train` method:
   ```
   results = model.train(
       data='data.yaml',
       epochs=35,
       batch=8,
       name='yolov8n_custom_all'
   )
   ```
3. Adjust training parameters such as epochs, batch size, and model name as needed in the script.
  
4. Ensure that the `data.yaml` file contains information about the dataset, including the class labels. In this project, there is only one class, which is 'fire'.

## Making Predictions:
To make predictions using the trained YOLOv8 model, follow these steps:

1. Load the YOLOv8 model with the trained weights:
   ```
   from ultralytics import YOLO
   model = YOLO("model/weights/best.pt")
   ```

2. Start capturing video from the webcam using OpenCV:
   ```
   cap = cv2.VideoCapture(0)
   ```

3. Extract frames from the video feed and perform object detection using the YOLOv8 model:
   ```
   ret, frame = cap.read()
   results = model.predict(source=frame, line_width=1, show_conf=False)
   ```

4. Draw bounding boxes around detected fires and display a "Danger: Fire Detected" message on the video feed:
   ```
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
   ```

5. Display the processed frame with bounding boxes and messages using OpenCV:
   ```
   cv2.imshow('Fire Detection', frame)
   ```

6. Press 'q' to exit the video feed.
