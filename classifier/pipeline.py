from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin

import pandas as pd

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
from processing import preprocess as pre
from processing import feature_extraction as fe
from processing import classifier as clf
import logging
_logger = logging.getLogger(__name__)
# --------------------------------------------------


# the pipeline steps including preprocessing, feature extraction , classifier
# GRU classifier is used but other calssifiers can be added too.

clfpip=Pipeline(steps=[("prerocess",pre.text_preprocess()),("feature_extraction",fe.feature_extraction()),("gru",clf.Gru())])


