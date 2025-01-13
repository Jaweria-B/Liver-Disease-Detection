import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# Streamlit App Interface
st.set_page_config(page_title="Liver Detection Bot", page_icon="🤖", layout="centered")

st.title("🤖 Liver Detection Bot")
st.write("Upload an image for liver classification based on tongue analysis.")

# Load the YOLO model
model = YOLO("/stage-2/runs/detect/train/weights/best.pt")

# File uploader for image input
uploaded_file = st.file_uploader("Upload a tongue image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Tongue Image", use_container_width=True)

    # Process the image
    with st.spinner("Analyzing image..."):
        image = Image.open(uploaded_file)
        results = model.predict(source=image, save=False)

        # Parse results
        predictions = []
        for box in results[0].boxes:
            predictions.append({
                "class": model.names[int(box.cls)],  # Class name
                "confidence": float(box.conf),      # Confidence score
                "bbox": box.xyxy.tolist()          # Bounding box coordinates
            })
        
        # Generate a response
        if predictions:
            first_prediction = predictions[0]
            pred_class = first_prediction.get("class", "Unknown")
            confidence = first_prediction.get("confidence", 0)

            if pred_class == "left_region":
                message = (
                    f"A potential abnormality has been detected on the left side of the tongue with "
                    f"{confidence * 100:.2f}% confidence. This might be indicative of liver disease. "
                    "We recommend consulting a healthcare provider for further examination."
                )
            elif pred_class == "right_region":
                message = (
                    f"A potential abnormality has been detected on the right side of the tongue with "
                    f"{confidence * 100:.2f}% confidence. This might suggest a liver-related issue. "
                    "Please seek medical advice for proper diagnosis."
                )
            elif pred_class == "discolored_patch":
                message = (
                    "The tongue appears to have a discolored or cracked surface, which is often associated with "
                    "symptoms like fatty liver disease. It’s advisable to consult a specialist."
                )
            elif pred_class == "normal_tongue":
                message = "The tongue appears normal with no signs of abnormalities. No liver disease is detected."
            else:
                message = "The analysis could not conclusively classify the tongue. Please try again with a clearer image."
            
            with st.chat_message("assistant"):
                st.success(message)
            # st.json(predictions)
        else:
            st.warning("The analysis could not conclusively classify the tongue. Please try again with a clearer image.")
