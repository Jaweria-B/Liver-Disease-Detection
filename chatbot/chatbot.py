import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Streamlit App Interface
st.set_page_config(page_title="Liver Detection Bot", page_icon="🤖", layout="centered")

st.title("🤖 Liver Detection Bot")
st.write("Ask me to analyze an image for liver classification. Upload an image and ask your question!")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("../stage-1/liver_classification_model.h5")

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to model's expected input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    class_names = ["Not Liver", "Liver"]  # Update based on your class indices
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]
    
    # Revert prediction if confidence is less than 50%
    if confidence < 0.40:
        if predicted_class == "Not Liver": 
            predicted_class = "Liver" 
    
    return predicted_class

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to display messages
def display_messages():
    for message in st.session_state.messages:
        if message["type"] == "bot":
            with st.chat_message("assistant"):
                # st.image(bot_avatar, width=40)  # Show bot avatar
                st.markdown(message["content"])
        else:
            with st.chat_message("user"):
                # st.image(user_avatar, width=40)  # Show user avatar
                st.markdown(message["content"])



# Display previous messages
display_messages()

# File uploader for image at the top
with st.sidebar:
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# User input
user_query = st.chat_input("Ask me about the uploaded image.")

if user_query:
    # Save user message
    st.session_state.messages.append({"type": "user", "content": user_query})

    if not uploaded_image:
        bot_response = "Please upload an image first so I can analyze it."
    else:
        # Load and display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Bot response with spinner
        with st.spinner("Analyzing the image..."):
            predicted_class = predict(image)
            if predicted_class == "Liver":
                bot_response = "The image is classified as **Liver Disease affected**."
            else:
                bot_response = "The image is classified as **not affected with Liver Disease**."


    # Save bot response
    st.session_state.messages.append({"type": "bot", "content": bot_response})

# Display all messages (only once at the end)
display_messages()


