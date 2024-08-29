import os
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from consonance.ml_logic.registry import load_model
from consonance.ml_logic.preprocessor import image_preprocess
from consonance.ml_logic.decoder import decode_predictions
from consonance.utils.converter import create_music_files

from fastapi.responses import JSONResponse
import base64
import cv2
import numpy as np
import io
from PIL import Image

app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        # key: str = "C",  # Default key
        # tempo: int = 120  # Default tempo
    ):
    """
    Generate a music file based on the provided data.
    """
    # Ensure the model is loaded
    assert app.state.model is not None

    # Read the image file
    image_data = await image.read()

    # Preprocess the image
    X_processed = image_preprocess(image_data)
    # print("\n🍌 X_processed type: 🍌 ", type(y_pred), "\n")

    # Make the prediction (this is a placeholder, TODO: replace with actual prediction logic)
    y_pred = app.state.model.predict(X_processed)
    # midi = generate_midi_placeholder_function(y_pred, format, key, tempo)  # TODO: convert prediction to MIDI format

    print("\n✅ prediction done:", y_pred, "\n")

    # return {'test_img': X_processed}
    # return {'midi': midi}

@app.post("/test_generate")
async def predict(
    request: Request,
    images: list[UploadFile] = File(...),
    media_type: str = Query("midi", enum=["midi", "wav", "mp3"]),
):
    """
    Process the image and return the preprocessed image.
    """
    # Read all image data and decode to NumPy arrays
    image_data_list = []
    for image in images:
        image_bytes = await image.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not decode image.")
        image_data_list.append(img)

    # Preprocess the array of images
    processed_images = image_preprocess(image_data_list)

    # # Convert to NumPy array and add channel dimension
    # processed_images = np.array(processed_images)
    # processed_images = processed_images[..., np.newaxis]  # Add channel dimension

    # Make the prediction
    y_pred = app.state.model.predict(processed_images)

    print("\n✅ Prediction done:", y_pred, "\n")
    # print("\n🍌 y_pred type: 🍌 ", type(y_pred), "\n")

    y_decoded = decode_predictions(y_pred)
    print("\n✅ Prediction y_decoded:\n")

    output_files = [create_music_files(note_string, media_type) for note_string in y_decoded]
    print("\n✅ Created output_files: 🍌", output_files, "\n")
    
    # # 1st try
    # files = []
    # for file in output_files:
    #     file_dict = {}
    #     if media_type in file:
    #         file_path = file[media_type]
    #         print("\n🍓 file_path: 🍓", file_path, "\n")
    #         file_dict[media_type] = FileResponse(file_path, media_type='application/octet-stream', filename=file[media_type])
    #     if media_type is not "wav":
    #         file_dict["wav"] = FileResponse(file_path, media_type='application/octet-stream', filename=file["wav"])
    #     files.append(file_dict)
    #     print("\n🍓 file_dict: 🍓", file_dict, "\n")

    # # 2nd try
    # file_responses = []
    # for file_dict in output_files:
    #     response_entry = {}
    #     if media_type in file_dict:
    #         file_path = file_dict[media_type]
    #         print("\n🍓 file_path: 🍓", file_path, "\n")
    #         response_entry[media_type] = FileResponse(file_path, media_type='application/octet-stream', filename=file_path)

    #     if media_type != "wav" and "wav" in file_dict:
    #         wav_path = file_dict["wav"]
    #         print("\n🍓 wav_path: 🍓", wav_path, "\n")
    #         response_entry["wav"] = FileResponse(wav_path, media_type='application/octet-stream', filename=wav_path)
    #     file_responses.append(response_entry)
    #     print("\n🍓 response_entry: 🍓", response_entry, "\n")
    
    # return file_responses


    # Create URLs to serve files
    file_urls = []
    for file_dict in output_files:
        file_info = {}
        for file_type, file_path in file_dict.items():
            if os.path.exists(file_path):
                # Generate the URL to download the file
                file_info[file_type] = str(request.url_for('download_file', file_path=file_path))
            else:
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        file_urls.append(file_info)
    print("\n🍓 file_urls: 🍓", file_urls, "\n")
    return file_urls


@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='application/octet-stream', filename=os.path.basename(file_path))


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
