import cv2
import glob
import numpy as np
import os
import pandas as pd

from consonance.utils.generate import generate_synthetic_single_musicxml, convert_musicxml_to_png

def generate_data():
    """generate training dataset"""
    generate_synthetic_single_musicxml(num_samples=500)
    convert_musicxml_to_png()

# def load_images_from_folder(folder):
#     """load images"""
#     images = []
#     for filename in glob.glob(f'{folder}/*.png'):
#         img = cv2.imread(filename)
#         if img is not None:
#             images.append(img)
#     return images

def load_images_with_filenames(folder):
    """load images with file names"""
    images = []
    filenames = []
    for filename in glob.glob(f'{folder}/*.png'):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames
        
def save_images_to_folder(images, folder, original_filenames):
    for img, original_filename in zip(images, original_filenames):
        base_filename = os.path.basename(original_filename)
        new_filename = os.path.join(folder, base_filename)
        cv2.imwrite(new_filename, img)


def load_labels(label_file='../raw_data/labels.csv'):
    labels_df = pd.read_csv(label_file)
    return labels_df.set_index('filename').to_dict()['label']

def create_single_note_dataset(image_folder='../raw_data/cropped_images', label_file='../raw_data/labels.csv'):
    ''''''
    labels = load_labels(label_file)
    images = []
    bounding_boxes = []
    image_labels = []

    for file_name in os.listdir(image_folder):
        if file_name.endswith('.png'):
            img_path = os.path.join(image_folder, file_name)
            label = labels[file_name]
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Create a bounding box covering the entire image
            h, w = img_array.shape
            bounding_box = [0, 0, w, h]

            images.append(img_array)
            bounding_boxes.append(bounding_box)
            image_labels.append(label)

    return np.array(images), bounding_boxes, image_labels

#### TODO: This will need to be adjusted (img --> file) #####
def create_dataset(num_samples):
    ''' For rows of music, TODO: to be tried later'''
    images = []
    labels = []
    for i in range(num_samples):
        img = cv2.imread(f'random_sample_{i}.png', cv2.IMREAD_GRAYSCALE)
        img_array = np.array(img)

        # Example bounding box creation (this should be based on actual note positions)
        bounding_boxes = [(50, 50, 100, 100)]  # Placeholder
        label = ['C4']  # Placeholder

        images.append(img_array)
        labels.append((bounding_boxes, label))

    return np.array(images), labels

# # DONT NEED AFTER ALL?
# def resize_with_aspect_ratio(img, target_size):
#     h, w = img.shape

#     # Calculate the aspect ratio
#     aspect_ratio = w / h

#     # Determine the target width and height based on the target size
#     if aspect_ratio > 1:  # Wider image
#         new_w = target_size[0]
#         new_h = int(target_size[0] / aspect_ratio)
#     else:  # Taller image
#         new_h = target_size[1]
#         new_w = int(target_size[1] * aspect_ratio)

#     # Resize the image
#     resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

#     # Add padding to make the image square
#     delta_w = target_size[0] - new_w
#     delta_h = target_size[1] - new_h
#     top, bottom = delta_h // 2, delta_h - (delta_h // 2)
#     left, right = delta_w // 2, delta_w - (delta_w // 2)

#     color = [255]  # Assuming a white background (255 for grayscale)
#     padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

#     return padded_img
