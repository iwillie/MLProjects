from sklearn.pipeline import Pipeline
import prediction_model.processing.preprocessing as pp
from prediction_model.config import config
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# classification_pipeline = Pipeline(
#     [
#         ("MeanInputation", pp.MeanImputer(variables=config.NUM_FEATURES)),
#         ("Modeimputation", pp.ModeImputer(variables=config.CAT_FEATURES)),
#         ("DomainProcessing", pp.DomainProcessing(variable_to_add=config.FEATURE_TO_ADD, variable_to_modify=config.FEATURE_TO_MODIFY)),
#         ("DropColumns", pp.DropColumns(variables_to_drop=config.DROPED_FEATURES)),
#         ("LabelEncoder", pp.CustomLabelEncoder(variables=config.CAT_FEATURES)),
#         ("LogTransform", pp.LogTransforms(variables=config.NUM_FEATURES)),
#         ("MinMaxScaler", MinMaxScaler()),
#         ("LogisticClassifier", LogisticRegression(random_state=0))
#     ]
# )

classification_pipeline = Pipeline(
    [
        ('DomainProcessing',pp.DomainProcessing(variable_to_modify = config.FEATURE_TO_MODIFY,
        variable_to_add = config.FEATURE_TO_ADD)),
        ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('ModeImputation',pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROPED_FEATURES)),
        ('LabelEncoder',pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransform',pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticClassifier',LogisticRegression(random_state=0))
    ]
)














