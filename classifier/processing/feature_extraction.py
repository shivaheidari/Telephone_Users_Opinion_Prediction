

from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin
from gensim.models import Word2Vec
import numpy as np

#  we aim to extract vectors of words. vectors are achived from learning vectors. vectores are learned based on cocurence
# of words

# each comment consists of words , the vectors are mapped to one vector from summation of word-vectors

class feature_extraction(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.model=[]
        print("fe")

    def fit(self,X,y=None):

      return self

    def transform(self,X):
        # learning vectors

        sent = [row.split(" ") for row in X]
        model=Word2Vec(sent,size=50,window=4,min_count=1,sg=1)
        for i,r in X.iteritems():
            sentence=r.split(" ")
            vector=[]
            sum_vec = np.zeros(50)
            for w in sentence:
                vector.append(model[w])
                sum_vec=sum_vec+model[w]
            X[i]=sum_vec


        return X
