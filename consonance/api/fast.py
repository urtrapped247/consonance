import os
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

media_types = {
        'midi': 'audio/midi',
        'wav': 'audio/wav',
        'mp3': 'audio/mpeg'
    }

@app.post("/generate")  # Use POST because we're uploading a file
async def predict(
        image: UploadFile = File(...),  # Accept image files
        media_type: str = Query("midi", enum=["midi", "wav", "mp3"]),  # Default to MIDI
        key: str = "C",  # Default key
        tempo: int = 120  # Default tempo
    ):      
    """
    Generate a music file based on the provided data.
    """
    
    # Read the image file
    image_data = await image.read()

    # Preprocess the image
    X_processed = image_preprocess(image_data)

    # Ensure the model is loaded
    assert app.state.model is not None

    # Make the prediction (this is a placeholder, TODO: replace with actual prediction logic)
    y_pred = app.state.model.predict(X_processed)
    midi = generate_midi_placeholder_function(y_pred, format, key, tempo)  # TODO: convert prediction to MIDI format

    print("\n✅ prediction done:", y_pred, "\n")
    
    return {'midi': midi}

@app.get("/test")
async def get_dummy_midi(media_type: str = Query(..., description="The media type of the file: 'midi', 'wav', or 'mp3'")):
    """
    Return a dummy file based on the provided media_type.
    """
    media_types = {
        'midi': 'audio/midi',
        'wav': 'audio/wav',
        'mp3': 'audio/mpeg'
    }
    
    if media_type in media_types:
        file_path = os.path.join(os.getcwd(), "consonance/api/hot-cross-buns.mid")
        print("file_path=\n", file_path)
        
        if not os.path.exists(file_path):
            # return {'error': f"File not found: {file_path}"}
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        file_name = "hot-cross-buns.mid"
        return FileResponse(file_path, media_type=media_types[media_type], filename=file_name)
    
    else:
        # return {'error': "Invalid media type. Please use 'midi', 'wav', or 'mp3'."}
        raise HTTPException(status_code=400, detail="Invalid media type. Please use 'midi', 'wav', or 'mp3'.")

@app.get("/")
def root():
    return {'greeting': 'Hello'}
