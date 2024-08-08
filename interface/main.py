''' '''

import cv2


from ml_logic.preprocessor import preprocessor

def preprocess() -> None:
    ''' '''
    
    # Load images
    ## 
    def load_images_from_folder(folder):
        images = []
        for filename in glob.glob(f'{folder}/*.png'):
            img = cv2.imread(filename)
            if img is not None:
                images.append(img)
        return images

    # Load images
    folder = 'path/to/your/image/folder'
    X = load_images_from_folder(folder)