from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
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
    
    # Extract the first prediction
    if predictions:
        first_prediction = predictions[0]
        pred_class = first_prediction.get("class", "Unknown")
        confidence = first_prediction.get("confidence", 0)

        # Generate a customized response based on the prediction
        response = {}

        if pred_class == "left_region":
            response["message"] = (
                f"A potential abnormality has been detected on the left side of the tongue with "
                f"{confidence * 100:.2f}% confidence. This might be indicative of liver disease. "
                "We recommend consulting a healthcare provider for further examination."
            )
        elif pred_class == "right_region":
            response["message"] = (
                f"A potential abnormality has been detected on the right side of the tongue with "
                f"{confidence * 100:.2f}% confidence. This might suggest a liver-related issue. "
                "Please seek medical advice for proper diagnosis."
            )
        elif pred_class == "dispatched_color":
            response["message"] = (
                "The tongue appears to have a discolored or cracked surface, which is often associated with "
                "symptoms like fatty liver disease. It’s advisable to consult a specialist."
            )
        elif pred_class == "normal_tongue":
            response["message"] = (
                "The tongue appears normal with no signs of abnormalities. No liver disease is detected."
            )
        else:
            response["message"] = (
                "The analysis could not conclusively classify the tongue. Please try again with a clearer image."
            )

        response["predictions"] = predictions
        return JSONResponse(content=response)

    else:
        return JSONResponse(content={"message": "No predictions found. Please try again."})