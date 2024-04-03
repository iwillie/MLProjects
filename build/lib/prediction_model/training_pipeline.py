import pandas as pd
import numpy as np
from prediction_model.config import config
from prediction_model.processing.datahandling import load_dataset, save_model
from prediction_model.processing import preprocessing as pp
import prediction_model.pipline as pipe
import sys

def perform_training():
    
    train_data = load_dataset(config.TRAIN_FILE)
    train_y = train_data[config.TARGET].map({"N": 0, "Y": 1})
    pipe.classification_pipeline.fit(train_data[config.FEATURES], train_y)
    save_model(pipe.classification_pipeline)
    
    
if __name__=='__main__':
    perform_training()














