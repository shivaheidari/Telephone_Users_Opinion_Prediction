from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin

import pandas as pd
from keras.models import Sequential
from keras.layers import GRU,Dense
from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing import LabelEncoder

# the GRU classifier is proposed for this problem.



# for defining  custom classifier ClassifierMixin, BaseEstimator should be used.
class Gru(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model=None
        self.label=[]
    def fit(self,X,y):

        #  encodigng lableling in the fit method, for mapping classes to vectors that will be fed to GRU
        # ie : Yes [1 0] ,No[0 1]
        encoder=LabelEncoder()
        encoder.fit(y)
        encoded_y=encoder.transform(y)
        dummy_y=np_utils.to_categorical(encoded_y)
        self.label=dummy_y


        #  making a cube for GRU. GRU input should be in the cube format and then it is fed to the classifier.
        #  Cube(number of smaples , number of GRU unites, the lenght of vector)
        X_=X.to_numpy()
        for i in range(0,X_.shape[0]):
            print(X_[i])
        out = np.concatenate(X_).ravel()
        print(out)
        X_ = out.reshape(X_.shape[0], 1, 50)
        # print(X_)
        # model consists of a GRU unit and a final unit for classification(softmax activation function is used
        # but tnh can be used also)
        model = Sequential()
        model.add(GRU(2, input_shape=(1, 50), return_sequences=False))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
        history=model.fit(X_,self.label,epochs=5)
        self.model=model
        return self
    def predict(self,X):
            #reshping the input is needed
            # making the cube for the classifier
            #  Cube(number of smaples , number of GRU unites, the lenght of vector)
            for i in range(0, X.shape[0]):
                print(X[i])
            out = np.concatenate(X).ravel()
            X = out.reshape(X.shape[0], 1, 50)
            model=self.model
            prediction=model.predict(X)
            predict_class = model.predict_classes(X)
            return predict_class