from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
import requests
import uvicorn

app = FastAPI()

# Load TFLite model and allocate tensors
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading model: {e}")

# Get input and output tensors
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Define optimal thresholds for each class
optimal_thresholds = {
    0: 0.9999739,  # Threshold for COVID
    1: 3.8397732e-18,  # Threshold for Normal
    2: 3.6775113e-05  # Threshold for Pneumonia
}

# Function to preprocess images
def preprocess_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image = image.resize((256, 256))  # Adjust to match model's input size
        image = np.array(image).astype(np.float32)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize to [0, 1]
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

# Function to get predictions
def get_prediction(image_bytes):
    image = preprocess_image(image_bytes)
    try:
        interpreter.set_tensor(input_details['index'], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        return output_data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Function to classify based on thresholds
def classify(predictions):
    class_names = ['covid', 'normal', 'pneumonia']
    results = {}
    max_class = None
    max_score = -1
    
    for i, score in enumerate(predictions):
        if score >= optimal_thresholds[i]:
            results[class_names[i]] = score
            if score > max_score:
                max_score = score
                max_class = class_names[i]
        else:
            results[class_names[i]] = 0
    
    if max_class:
        return f"You have {max_class}"
    else:
        return "No clear diagnosis"

# Endpoint to handle file uploads
@app.post("/predict/file/")
async def predict_file(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        predictions = get_prediction(image_bytes)
        result_message = classify(predictions)
        return {"message": result_message}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Model to validate URL input
class ImageURL(BaseModel):
    url: str

# Endpoint to handle image URL
@app.post("/predict/url/")
async def predict_url(image_url: ImageURL):
    try:
        response = requests.get(image_url.url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download image from URL")
        image_bytes = response.content
        predictions = get_prediction(image_bytes)
        result_message = classify(predictions)
        return {"message": result_message}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
