import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from consonance.ml_logic.registry import load_model
from consonance.ml_logic.preprocessor import image_preprocess

app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/generate")  # Use POST because we're uploading a file
async def predict(
        image: UploadFile = File(...),  # Accept image files
        key: str = "C",  # Default key
        tempo: int = 120  # Default tempo
    ):      
    """
    Generate a MIDI file based on the provided data.
    """
    
    # Read the image file
    image_data = await image.read()

    # Preprocess the image
    X_processed = image_preprocess(image_data)

    # Ensure the model is loaded
    assert app.state.model is not None

    # Make the prediction (this is a placeholder, TODO: replace with actual prediction logic)
    y_pred = app.state.model.predict(X_processed)
    midi = y_pred  # TODO: convert prediction to MIDI format

    print("\n✅ prediction done:", y_pred, "\n")
    
    return {'midi': midi}

# @app.get("/")
# def root():
#     return {'greeting': 'Hello'}
