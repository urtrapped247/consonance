''' '''
import cv2

from consonance.ml_logic.data import generate_data, load_images_from_folder
from consonance.ml_logic.preprocessor import image_preprocess, crop_note_from_png_folder
from consonance.utils.utils import is_directory_empty

def preprocess() -> None:
    '''preprocess images'''

    sheet_folder = '../raw_data/sheet_images'
    cropped_folder = '../raw_data/cropped_images'
    
    # get images from raw_data folder
    if is_directory_empty(sheet_folder):
        print(f"The directory {sheet_folder} is empty.")
        generate_data()
        
    #  cropp images
    crop_note_from_png_folder(sheet_folder, cropped_folder)
        
    # load images
    X = load_images_from_folder(cropped_folder)
    
    # Process data
    X_processed = image_preprocess(X)
    