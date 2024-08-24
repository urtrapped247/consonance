import cv2
import glob
import numpy as np
import os
import random
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from keras import layers
from omr import *

def image_preprocess(X) -> np.ndarray:
    class GrayscaleTransformer(BaseEstimator, TransformerMixin):
        '''Converts images to grayscale.'''
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X]

    class NoiseReducer(BaseEstimator, TransformerMixin):
        '''Applies Gaussian blur to reduce noise.'''
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return [cv2.GaussianBlur(img, (5, 5), 0) for img in X]

    class Binarizer(BaseEstimator, TransformerMixin):
        '''Converts images to binary format using Otsu’s thresholding.'''
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return [cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] for img in X]

    class Augmenter(BaseEstimator, TransformerMixin):
        '''Applies augmentation techniques including rotation, scaling, translation, shearing, noise addition, and blurring.'''
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return [self.augment_image(img) for img in X]

        def augment_image(self, image):
            rows, cols = image.shape

            # Rotation
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(image, M, (cols, rows))

            # Scaling
            scale = random.uniform(0.9, 1.1)
            resized = cv2.resize(rotated, None, fx=scale, fy=scale)

            # Translation
            tx = random.randint(-5, 5)
            ty = random.randint(-5, 5)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            translated = cv2.warpAffine(resized, M, (cols, rows))

            # Shearing
            shear = random.uniform(-0.1, 0.1)
            M = np.float32([[1, shear, 0], [0, 1, 0]])
            sheared = cv2.warpAffine(translated, M, (cols, rows))

            # Noise addition
            noise = np.random.randint(0, 50, (rows, cols), dtype='uint8')
            noisy = cv2.add(sheared, noise)

            # Blur
            blurred = cv2.GaussianBlur(noisy, (5, 5), 0)

            return blurred

    # Combine into a pipeline
    image_preprocessor = Pipeline([
        ('grayscale', GrayscaleTransformer()),
        ('denoise', NoiseReducer()),
        ('binarize', Binarizer()),
        ('augment', Augmenter())
    ])

    # Preprocess images
    processed_images = image_preprocessor.fit_transform(X)

    return processed_images

def crop_note_from_png_folder(input_folder, output_folder):
    """
    Crops all PNG images in the specified folder to the specified dimensions.

    Parameters:
    - input_folder (str): The path to the input folder containing PNG images.
    - output_folder (str): The path to the folder to save the cropped images.
    """
    # Define the crop box (left, upper, right, lower)
    crop_box = (506, 536, 580, 870)  # Replace these values with your desired dimensions

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image file
            with Image.open(input_path) as img:
                # Crop the image using the provided crop box
                cropped_img = img.crop(crop_box)

                # Save the cropped image
                cropped_img.save(output_path)

            # print(f'Cropped image saved to {output_path}')

# # Display original and processed images for comparison
# for i in range(5):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
#     plt.title('Original')

#     plt.subplot(2, 5, i+6)
#     plt.imshow(processed_images[i], cmap='gray')
#     plt.title('Processed')

# plt.show()

def resize_with_aspect_ratio(img, target_size):
    '''
    Code from notebook
    TODO: include resizing with padding in image_preprocess?
    '''
    h, w = img.shape

    # Calculate the aspect ratio
    aspect_ratio = w / h

    # Determine the target width and height based on the target size
    if aspect_ratio > 1:  # Wider image
        new_w = target_size[0]
        new_h = int(target_size[0] / aspect_ratio)
    else:  # Taller image
        new_h = target_size[1]
        new_w = int(target_size[1] * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Add padding to make the image square
    delta_w = target_size[0] - new_w
    delta_h = target_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255]  # Assuming a white background (255 for grayscale)
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_img


#put in model here
prediction_model = pass

# Preprocess the image
def preprocess_image(image_path):
    img_width = 500
    img_height = 50
    # Read the image
    img = tf.io.read_file(image_path)
    # Decode the image and convert it to grayscale
    img = tf.io.decode_png(img, channels=1)
    # Resize the image to the expected size
    img = tf.image.resize(img, [img_height, img_width])
    # Normalize the image to the range [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Transpose the image to match the input shape
    img = tf.transpose(img, perm=[1, 0, 2])
    # Add batch dimension
    img = tf.expand_dims(img, axis=0)
    return img

# Predict text from image
def predict_text_and_display(prediction_model, image_path):
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    # Make predictions
    predictions = prediction_model.predict(processed_img)
    # Decode the predictions
    decoded_text = decode_batch_predictions(predictions)

    return decoded_text


image_path = "/Users/ninjamac/code/images_cropped/music_99.png"  # Replace with the actual path to your image
