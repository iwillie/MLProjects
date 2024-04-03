import pytest
from prediction_model.config import config
from prediction_model.processing.datahandling import load_dataset
from prediction_model.predict import generate_prediction

# output from predict not null
# output from predict script is of str data type
# output is Y from example data

@pytest.fixture
def single_prediction():
    test_data = load_dataset(config.TEST_FILE)
    single_row = test_data[:1]
    result = generate_prediction(single_row)
    return result


def test_pred_not_none(single_prediction):
    assert single_prediction is not None
    
def test_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get("prediction")[0], str)
    
def test_pred_validate(single_prediction):
    assert single_prediction.get("prediction")[0] == "Y"
