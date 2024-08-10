import cv2
import glob
from consonance.utils.generate import generate_synthetic_single_musicxml, convert_musicxml_to_png

def generate_data():
    """generate training dataset"""
    generate_synthetic_single_musicxml(num_samples=500)
    convert_musicxml_to_png()

def load_images_from_folder(folder):
    """load images"""
    images = []
    for filename in glob.glob(f'{folder}/*.png'):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images
