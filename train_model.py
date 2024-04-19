from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt')

# Training.
results = model.train(
    data='data.yaml',
    epochs=35,
    batch=8,
    name='yolov8n_custom_all'
)