from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Pretrained YOLOv8n model

# --------- Path Validation ---------
# model.val(data="./data.yaml")

# --------- Model Training ---------
# model.train(data="./data.yaml", epochs=50, imgsz=640, batch=16)

# Load the trained model (use the best.pt file after training)
model = YOLO("./runs/detect/train/weights/best.pt")

# # Test the model on a single image
image_path = "./dataset/1-s2.0-S0973688321005995-gr8.jpg"  # Replace with the path to your test image
results = model.predict(source=image_path, save=True, save_txt=True, conf=0.25)

# # Access the first result (since predict returns a list)
result = results[0]

# # Display the result
result.plot()  # This generates the output image with bounding boxes
