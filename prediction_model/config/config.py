import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")
MODEL_NAME = "classification.pkl"

TARGET = "Loan_Status"

# features used in model
FEATURES = ['Gender', 'Married', 'Dependents', 'ApplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area', 'CoapplicantIncome']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
CAT_FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

# columns to encode same as CAT_FEATURES
FEATURES_TO_ENCODE = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

FEATURE_TO_ADD = 'CoapplicantIncome'
FEATURE_TO_MODIFY = ['ApplicantIncome']
DROPED_FEATURES = 'CoapplicantIncome'

LOG_FEATURES = ['ApplicantIncome', 'LoanAmount']
