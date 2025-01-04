# Liver Detection Bot 🤖

This **Liver Detection Bot** is a Streamlit app designed to analyze images for liver classification. Users can upload an image (specifically of the tongue) and receive predictions from a backend model through FastAPI. This project leverages machine learning models to assess the image and provide feedback related to liver conditions.

## Features

- Upload an image of the tongue for analysis.
- The app sends the uploaded image to a FastAPI endpoint for liver classification.
- Displays the result of the liver condition based on the image analysis.
- Easy-to-use Streamlit interface.

## Requirements

Make sure you have the following dependencies installed:

- **Python 3.x**
- **Streamlit**: For building the web interface.
- **FastAPI**: For serving the model.
- **Requests**: For making HTTP requests to the FastAPI server.

Install required Python packages by running:

```bash
pip install streamlit fastapi requests
```

## Running the App

1. **Start the FastAPI server**:
    - Navigate to the folder where your FastAPI app is located and run:

    ```bash
    uvicorn app:app --reload
    ```

2. **Run the Streamlit app**:
    - Navigate to the folder containing your Streamlit app and run:

    ```bash
    streamlit run liver_detection_bot.py
    ```

    This will start a local Streamlit server, and the app will be accessible at `http://localhost:8501`.

3. **Upload an Image**:
    - Once the Streamlit app is running, you can upload an image of the tongue. The app will then process the image and display the result.

## API Endpoints

The FastAPI server provides an endpoint for liver detection:

- **POST `/predict/`**: Accepts an image file and returns a JSON response containing the liver classification result.

## Project Structure

- `app.py`: Main Streamlit app file.
- `main.py`: FastAPI app that processes the image and provides predictions.
- `requirements.txt`: List of Python dependencies for the project.

## Contributing

Feel free to fork the repository and contribute improvements! Make sure to create a pull request for any changes you want to propose.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
