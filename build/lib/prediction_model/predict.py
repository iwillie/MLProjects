import pandas as pd
import numpy as np
import joblib
from prediction_model.config import config
from prediction_model.processing.datahandling import load_model, load_dataset

classification_model = load_model(config.MODEL_NAME)

def generate_prediction(input_data):
    data = pd.DataFrame(input_data)
    pred = classification_model.predict(data[config.FEATURES])
    output = np.where(pred == 1, "Y", "N")
    result = {"prediction": output}
    return result


# def generate_prediction():
#     test_data = load_dataset(config.TEST_FILE)
#     pred = classification_model.predict(test_data[config.FEATURES])
#     output = np.where(pred == 1, "Y", "N")
#     # result = {"prediction": output}
#     print(output)
#     return output


# if __name__ == "__main__":
#     generate_prediction(input_data)









