#File: sentiment_mod.py

import nltk
import random

from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


import warnings
warnings.filterwarnings("ignore")


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = float(choice_votes) / len(votes)
        return conf


documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()




word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = set(word_tokenize(document))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


class_list_mod=[]
##featuresets_f = open("pickled_algos/featuresets.pickle", "rb")
##featuresets = pickle.load(featuresets_f)
##featuresets_f.close()
##
##random.shuffle(featuresets)
##print("no of features  "+str(len(featuresets)))
##
##training_set = featuresets[1000:2000]
##testing_set = featuresets[10000:]
## 

open_file = open("pickled_algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()
class_list_mod.append(classifier)
del open_file
del classifier
##print(nltk.classify.accuracy(classifier,testing_set)*100)


open_file = open("pickled_algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()
class_list_mod.append(MNB_classifier)
del open_file
del MNB_classifier
##print("Multinomial NAIVE BAYES ACCURACY:",nltk.classify.accuracy(MNB_classifier,testing_set)*100)



open_file = open("pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()
class_list_mod.append(BernoulliNB_classifier)
del open_file
del BernoulliNB_classifier
##print("BERNOULLI NAIVE BAYES ACCURACY:",nltk.classify.accuracy(BernoulliNB_classifier,testing_set)*100)



open_file = open("pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()
class_list_mod.append(LogisticRegression_classifier)
del open_file
del LogisticRegression_classifier
##print("Logistic Regression ACCURACY:",nltk.classify.accuracy(LogisticRegression_classifier,testing_set)*100)


open_file = open("pickled_algos/SGD_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()
class_list_mod.append(SGDC_classifier)
del open_file
del SGDC_classifier
##print("Stochastic Gradient ACCURACY:",nltk.classify.accuracy(SGDC_classifier,testing_set)*100)



open_file = open("pickled_algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()
class_list_mod.append(LinearSVC_classifier)
del open_file
del LinearSVC_classifier
##print("Linear SVC ACCURACY:",nltk.classify.accuracy(LinearSVC_classifier,testing_set)*100)


open_file = open("pickled_algos/NUSVC_classifier5k.pickle", "rb")
NSVC_classifier = pickle.load(open_file)
open_file.close()
class_list_mod.append(NSVC_classifier)
del open_file
del NSVC_classifier
##print("NSVC ACCURACY:",nltk.classify.accuracy(NSVC_classifier,testing_set)*100)






##voted_classifier = VoteClassifier(
##                                  classifier,
##                                  NSVC_classifier,
##                                  LinearSVC_classifier,
##                                  MNB_classifier,
##                                  SGDC_classifier,
##                                  BernoulliNB_classifier,
##                                  LogisticRegression_classifier)



##print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
    feats = find_features(text)
    votes=[]
    for cl in class_list_mod:
        votes.append(cl.classify(feats))
    choice_votes = votes.count(mode(votes))
    conf = float(choice_votes) / len(votes)
    print("voted_classifier" , mode(votes) , "with confidence percent:" , conf*100)
    
############################  CONFIDENCE MEASURE OF TESTING SET     ########################################
##def show():
##    
##    for i in range(len(testing_set)):
##        print(documents[i][0])    
##        conf=voted_classifier.confidence(testing_set[i][0])
##        v=voted_classifier.classify(testing_set[i][0])
##        print(v,"==",conf*100)

#############################################################################################################

##sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!")
