# all tasks with the data and pipeline should be add here.
# save pipeling, load dataset ,save pipeline and removing old pipeliens
import joblib
import logging
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import typing as t
import os,sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from classifier import __version__ as _version

_logger = logging.getLogger(__name__)


# saving pipeline
# for saving a new pipeline, the old pipeline should be removed.
# pipelines are saved in a pkl file
def save_pipeline(*, pipeline_to_persist) -> None:

    save_file_name="clf.pkl"
    remove_old_pipelines(file_to_keep=[save_file_name])
    save_path="../trained_model/"+save_file_name
    joblib.dump(pipeline_to_persist,save_path)

def load_dataset(filename):

    _data=pd.read_csv(f'{config.DATASET_DIR}/{filename}', sep=';',skiprows=1,names=["NUM", "INTERACTION_ID", "INTERACTION_SUBJECT", "SALES_AREA_EN_DESC", "CRTD_USR_ID",
                              "CRTD_DT", "BUS_PRTNR_ID", "INTERACTION_TYP_ID", "INTERACTION_STYP_ID",
                              "INTERACTION_NOTE_ID", "Roger's comments", "label", "INTERACTION_COMMENT"])
    _data = _data.loc[_data["label"].isin(["Yes", "No"])]
    ds_mth_2 = _data.groupby("INTERACTION_ID")
    ds_mth_2 = ds_mth_2.filter(lambda x: len(x) >= 2)
# among records with more than one interaction_id, the old one is important.
    result = pd.merge(ds_mth_2, ds_mth_2, on=["INTERACTION_ID"])
    for i, r in result.iterrows():
        date_1 = str(r["CRTD_DT_x"])
        date_2 = str(r["CRTD_DT_y"])
        date_obj_1 = datetime.strptime(date_1, '%Y-%m-%d %H:%M:%S')
        date_obj_2 = datetime.strptime(date_2, '%Y-%m-%d %H:%M:%S')
        if (date_obj_1.date() > date_obj_2.date()):
            # delete the new one
            ds_mth_2 = ds_mth_2[ds_mth_2.INTERACTION_NOTE_ID != (r["INTERACTION_NOTE_ID_x"])]

    ds_fltr = _data
    for i, r in ds_mth_2.iterrows():
        ds_fltr = ds_fltr[ds_fltr.INTERACTION_ID != (r["INTERACTION_ID"])]
    _data = pd.concat([ds_fltr, ds_mth_2])
    _data = _data[_data["INTERACTION_COMMENT"].notnull()]
    return _data


def save_pipeline(pipeline_to_persist):
    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f'saved pipeline: {save_file_name}')


def load_pipeline(*, file_name: str
                  ) -> Pipeline:

    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()