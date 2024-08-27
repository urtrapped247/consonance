import glob
import os
import time
import pickle

from colorama import Fore, Style
import tensorflow as tf
from tensorflow import keras
# from google.cloud import storage

from consonance.params import *
# import mlflow
# from mlflow.tracking import MlflowClient

def save_results(params: dict, metrics: dict) -> None:
    """    """
    print("✅ Results saved locally")


def save_model(model: keras.Model = None) -> None:
    """    """
    return None


def load_model(stage="Production") -> keras.Model:
    """Return trained model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '../models/prediction_base_model.keras')
    production_model = tf.keras.models.load_model(model_path)
    return production_model
