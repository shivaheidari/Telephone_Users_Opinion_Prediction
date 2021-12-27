import os
import pathlib
#set  values and dirctions here


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_model'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

#  pipeline name comes here.
PIPELINE_NAME="classifier"
PIPELINE_SAVE_FILE=f'{PIPELINE_NAME}_output_v_'


# name of training and testing data in csv format
TESTING_DATA_FILE = 'interaction_360_for_labeling_15Jan20_AK.csv'
TRAINING_DATA_FILE = 'interaction_360_for_labeling_15Jan20_AK.csv'