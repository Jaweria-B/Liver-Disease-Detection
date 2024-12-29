import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("liver_classification_model.h5")

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
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Streamlit App Interface
st.set_page_config(page_title="Liver Detection Bot", page_icon="🤖", layout="centered")

st.title("🤖 Liver Detection Bot")
st.write("Ask me to analyze an image for liver classification. Upload an image and ask your question!")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Avatars
bot_avatar = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"
user_avatar = "https://cdn-icons-png.flaticon.com/512/1946/1946429.png"

# Function to display messages
def display_messages():
    for message in st.session_state.messages:
        if message["type"] == "bot":
            with st.chat_message("assistant", avatar_url=bot_avatar):
                st.markdown(message["content"])
        else:
            with st.chat_message("user", avatar_url=user_avatar):
                st.markdown(message["content"])

# Display previous messages
display_messages()

# File uploader for image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# User input
user_query = st.chat_input("Ask me about the uploaded image.")

if user_query:
    # Save user message
    st.session_state.messages.append({"type": "user", "content": user_query})
    display_messages()

    if not uploaded_image:
        with st.chat_message("assistant", avatar_url=bot_avatar):
            st.markdown("Please upload an image first so I can analyze it.")
    else:
        # Load and display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Bot response with spinner
        with st.chat_message("assistant", avatar_url=bot_avatar):
            with st.spinner("Analyzing the image..."):
                predicted_class, confidence = predict(image)
                response = f"Based on the analysis, the image is classified as **{predicted_class}** with a confidence of **{confidence:.2f}**."

                # Append response to messages
                st.session_state.messages.append({"type": "bot", "content": response})
                st.markdown(response)

        # Display updated messages
        display_messages()
