import numpy as np
import time

# Timing the import
print(Fore.BLUE + "\nLoading imports..." + Style.RESET_ALL)
start = time.perf_counter()

from colorama import Fore, Style
from tensorflow.keras import layers, models, Model, optimizers, regularizers, Sequential
# from keras.callbacks import EarlyStopping
# from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

end = time.perf_counter()
print(f"\n✅ imports loaded ({round(end - start, 2)}s)")

# # No need for this anymore TODO: confirm & delete
# def train_logistic_regression(X, y):
#     X_flat = X.reshape(X.shape[0], -1)
#     X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2)
#     model = LogisticRegression()
#     model.fit(X_train, y_train)
#     accuracy = model.score(X_test, y_test)
#     return accuracy

def create_cnn_model(input_shape, num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("✅ Model initialized")
    return model

def train_cnn(model, X, y):
    """
    Train the CNN model on the preprocessed dataset.
    TODO complete with code extracted from main.py
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = create_cnn_model(X_train.shape[1:])  # Example with 10 classes
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    return model.evaluate(X_test, y_test)


# def initialize_model(params) -> Model:
#     """
#     Initialize the Neural Network with random weights
#     """
    
# def compile_model(model: Model, learning_rate=0.0005) -> Model:
#     """
#     Compile the Neural Network
#     """
    
# def train_model(params): #-> Tuple[Model, dict]:
#     """
#     Fit the model and return ...
#     """

# def evaluate_model(
#         model: Model,
#         X: np.ndarray,
#         y: np.ndarray,
#         batch_size=64
#     ) -> Tuple[Model, dict]:
#     """
#     Evaluate trained model performance on the dataset
#     """
