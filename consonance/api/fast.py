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
    
@app.post("/test_generate")
async def predict(
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

    # Convert each processed image to base64
    processed_images_base64 = []
    for processed_img in processed_images:
        image_pil = Image.fromarray(processed_img.astype("uint8"))
        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        processed_images_base64.append(img_str)

    return JSONResponse(content={"processed_images": processed_images_base64})

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
