''' '''
import cv2
import tensorflow.keras as keras
import numpy as np
import pandas as pd

from consonance.ml_logic.model import create_cnn_model
from consonance.ml_logic.data import create_single_note_dataset, generate_data, load_images_with_filenames, save_images_to_folder
from consonance.ml_logic.preprocessor import crop_note_from_png_folder, image_preprocess
from consonance.utils.utils import is_directory_empty
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

def preprocess() -> None:
    '''preprocess images'''

    #  TODO: transfer to env variables
    sheet_folder = '../raw_data/sheet_images'
    cropped_folder = '../raw_data/cropped_images'
    preprocessed_folder = '../raw_data/preprocessed_images'
    
    # get images from raw_data folder
    if is_directory_empty(sheet_folder):
        print(f"The directory {sheet_folder} is empty.")
        generate_data()
        
    #  cropp images
    crop_note_from_png_folder(sheet_folder, cropped_folder)
        
    # load images
    X, original_filenames = load_images_with_filenames(cropped_folder)
    
    # Process data
    X_processed = image_preprocess(X)
    
    # save # Save the processed images to the preprocessed_folder with original filenames
    save_images_to_folder(X_processed, preprocessed_folder, original_filenames)
    # return X_processed

def train():
    ''' TODO: extract some logic to train_cnn() in model.py '''
    images, bounding_boxes, image_labels = create_single_note_dataset()
    
    X_processed = image_preprocess(images) ## TODO: refactor to use the preprocess function when model is ready
    num_classes = 10 # TODO param
    y = keras.utils.to_categorical(image_labels, num_classes=num_classes)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    model = create_cnn_model(X_train.shape[1:], num_classes=num_classes)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=20,  # Number of epochs, adjust as needed
        validation_data=(y_train, y_test),
        callbacks=[early_stopping]
    )
    
    evaluation = model.evaluate(X_test, y_test)
    
    return evaluation, history


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    TODO
    """