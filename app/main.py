from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch
from typing import List
from .src.model import cnnModel  # Assuming cnnModel is in cnn_model.py
from .src.utils import transform_image
# Initialize FastAPI app
app = FastAPI()

# Load the model from the checkpoint (.ckpt)
ckpt_path = "./weights/tf_efficientnet_b4.ns_jft_in1k_valid_loss=0.26302555203437805_valid_accuracy=0.9011653065681458.ckpt"  # Replace with actual checkpoint file
model = cnnModel(model_name="timm/tf_efficientnet_b4.ns_jft_in1k", num_classes=1)
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # Set model to evaluation mode

# Prediction function
def predict_image(image_tensor):
    with torch.no_grad():
        logits = model(image_tensor)
        prediction = torch.sigmoid(logits).item()  # Sigmoid for binary classification
    return prediction


# Default route
@app.get("/")
async def default_route():
    """
    Default route that returns a welcome message.
    """
    return {"message": "Welcome to the Pneumothorax Classification Project!"}


# API Endpoint: Handle multiple image predictions
@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    try:
        predictions = []
        
        # Iterate over each uploaded image
        for file in files:
            # Validate image file type
            if file.content_type not in ["image/jpeg", "image/png"]:
                raise HTTPException(status_code=400, detail=f"Invalid image type for file: {file.filename}")

            # Read and preprocess image
            image_bytes = await file.read()
            image_tensor = transform_image(image_bytes)

            # Run model prediction
            prediction = predict_image(image_tensor)

            # Append prediction results
            predictions.append({
                "filename": file.filename,
                "pneumothorax_prob": prediction,
                "diagnosis": "Pneumothorax" if prediction >= 0.5 else "No Pneumothorax"
            })

        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        # Return a meaningful error message
        return JSONResponse(status_code=500, content={"message": f"Error occurred: {str(e)}"})

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)