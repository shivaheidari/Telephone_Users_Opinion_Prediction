import sys,os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from classifier.processing.datamanagement import load_pipeline
from classifier.config import config
from  classifier import __version__ as _version
import logging
import typing as t



_logger = logging.getLogger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
clf = load_pipeline(file_name=pipeline_file_name)
# make prediction function is called for predicting based on predcition model that is saved in pkl file.
def make_prediction(input_data):

    # data = pd.read_csv(input_data,sep=';',skiprows=1,names=["NUM", "INTERACTION_ID", "INTERACTION_SUBJECT", "SALES_AREA_EN_DESC", "CRTD_USR_ID",
    #                           "CRTD_DT", "BUS_PRTNR_ID", "INTERACTION_TYP_ID", "INTERACTION_STYP_ID",
    #                           "INTERACTION_NOTE_ID", "Roger's comments", "label", "INTERACTION_COMMENT"])

    data=input_data
    data=(data["INTERACTION_COMMENT"].iloc[0])
    data=pd.Series(data)
    print(data)
    print(type(data))


    prediction = clf.predict(data)

    results = {'predictions': prediction, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {data} '
        f'Predictions: {results}')

    return results

# results = make_prediction(input_data="datasets/interaction_360_for_labeling_15Jan20_AK.csv")