import os
import joblib
import pandas as pd
from prediction_model.config import config

# Load dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data

# Serialization
def save_model(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved as {config.MODEL_NAME}")

# Deserialization
def load_model(model_to_load):
    save_path = os.path.join(config.SAVE_MODEL_PATH, model_to_load)
    loaded_model = joblib.load(save_path)
    print("Model has been loaded")
    return loaded_model
    