from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load the YOLO model
model = YOLO("../runs/detect/train/weights/best.pt")

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    # Read the uploaded image
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))

    # Run inference
    results = model.predict(source=image, save=False)

    # Parse results
    predictions = []
    for box in results[0].boxes:
        predictions.append({
            "class": model.names[int(box.cls)],  # Class name
            "confidence": float(box.conf),      # Confidence score
            "bbox": box.xyxy.tolist()          # Bounding box coordinates
        })
    
    return {"predictions": predictions}
