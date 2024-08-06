'''
Model related functions.
'''
import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

def initialize_model(params) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    
def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    
def train_model(params): #-> Tuple[Model, dict]:
    """
    Fit the model and return ...
    """

# def evaluate_model(
#         model: Model,
#         X: np.ndarray,
#         y: np.ndarray,
#         batch_size=64
#     ) -> Tuple[Model, dict]:
#     """
#     Evaluate trained model performance on the dataset
#     """
