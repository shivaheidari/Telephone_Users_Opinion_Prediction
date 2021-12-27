
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pipeline
import logging
from __init__ import __version__ as _version
from config import config
from processing.datamanagement import (load_dataset,save_pipeline)


_logger = logging.getLogger(__name__)

def run_training():
    data=load_dataset(filename=config.TRAINING_DATA_FILE)
    x_train, x_test, y_train, y_test = train_test_split(data["INTERACTION_COMMENT"], data["label"], train_size=0.8)
    pipeline.clfpip.fit(x_train,y_train)
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline.clfpip)

    return 0


if __name__=="__main__":
    run_training()

run_training()