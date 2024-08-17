import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from consonance.ml_logic.registry import load_model
from consonance.ml_logic.preprocessor import image_preprocess

# from dotenv import load_dotenv
# load_dotenv()

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

"""
params:
    image file
    key
    tempo
"""
@app.get("/generate")
def predict(
        image: file,       # what should be the data type?
        key: str,    # -73.950655
        tempo: int,     # 0-200
    ):      
    """
    Generate a midi file based on provided data.
    """
    
    X_pred = pd.DataFrame(dict(
        image=image,
        key=key,
        tempo=tempo,
    ))

    # model = load_model()
    assert app.state.model is not None

    # TODO: substitute with the actual fucntions responsible for this part when the model is complete:
    X_processed = image_preprocess(X_pred)
    y_pred = app.state.model.predict(X_processed)
    midi = y_pred # TODO: assign midi file (converted xml)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    print("\n✅ prediction done: ", fare_prediction, "\n")
    
    return {'midi': fare_prediction}


# @app.get("/")
# def root():
#     return {'greeting': 'Hello'}
