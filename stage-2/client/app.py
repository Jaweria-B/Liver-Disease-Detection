import streamlit as st
import requests


# Streamlit App Interface
st.set_page_config(page_title="Liver Detection Bot", page_icon="🤖", layout="centered")

st.title("🤖 Liver Detection Bot")
st.write("Ask me to analyze an image for liver classification. Upload an image and see the result!")

# File uploader for image input
uploaded_file = st.file_uploader("Upload a tongue image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Tongue Image", use_column_width=True)

    # Send image to FastAPI
    with st.spinner("Analyzing image..."):
        files = {"image": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict/", files={"image": uploaded_file})
    
    if response.status_code == 200:
        result = response.json()
        # st.success(result["message"])
        with st.chat_message("assistant"):
                # st.image(bot_avatar, width=40)  # Show bot avatar
                st.markdown(result["message"])

        # Optionally display predictions
        st.json(result["predictions"])
    else:
        st.error("Failed to process the image. Please try again.")
