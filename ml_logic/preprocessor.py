import cv2
import numpy as np
import glob
import random
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

def preprocess(X) -> np.ndarray:
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

# # Display original and processed images for comparison
# for i in range(5):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
#     plt.title('Original')
    
#     plt.subplot(2, 5, i+6)
#     plt.imshow(processed_images[i], cmap='gray')
#     plt.title('Processed')

# plt.show()
