from nltk.tokenize import TweetTokenizer
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin


# preprocessing of sentences . numbers, punctuation and,.. are imposing noises to the model. So, they should be eleminated.
class text_preprocess(BaseEstimator,TransformerMixin):


    def __init__(self):
        self.container=[]
        self.stop_words = set(stopwords.words('english'))
        self.lemmantizer = WordNetLemmatizer()
        self.ps = PorterStemmer()
        print("initiated")
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for i,r in X.iteritems():
            comment=r
            #call numeber removing
            comment=self.number_removing(comment)
            #call puncutation
            comment=self.punctuation(comment)
            #call tokenizing
            commnet=self.tokenizing(comment)
            #call stop word removing
            comment=self.stopword(commnet)
            # call lemmanization
            commnet=self.lemmatization(comment)
            #call stemming
            comment=self.stemming(comment)
            #call sentesize
            comment=self.sentesize(comment)
            X[i]=comment


        return X

    def tokenizing(self,commnet):
        sentence = re.sub(r"http\S+", "", commnet)
        tknzr = TweetTokenizer()
        tokenized = tknzr.tokenize(sentence)
        return tokenized
    def number_removing(self,comment):
        comment=re.sub(r'\d+','',comment)
        return comment

    def punctuation(self,comment):
        comment = re.sub('[' + string.punctuation + ']', '',comment)
        return comment

    def stopword(self,comment):
        filtered = []

        for w in comment:
            if w not in self.stop_words:
                filtered.append(w)

        return filtered

    def lemmatization(self,comment):

        lemma = []
        for w in comment:
            lemma.append(self.lemmantizer.lemmatize(w))
        return lemma

    def stemming(self,comment):

        stemed = []
        for w in comment:
            word = self.ps.stem(w)
            stemed.append(word)
        return stemed

    def sentesize(self,comment):
        comment = ' '.join(word for word in comment)
        return comment