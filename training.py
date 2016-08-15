from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
import time


import nltk
import random
import pickle
from statistics import mode

import warnings
warnings.filterwarnings("ignore")


############################   DATA ORGANIZATION FOR TWEETS ###################################################
short_pos=open("short/positive_1.txt","rb").read()
short_pos_sent=short_pos.split("\n")
del short_pos
short_neg=open("short/negative_1.txt","rb").read()
short_neg_sent=short_neg.split("\n")
del short_neg
documents=[]

for r in short_pos_sent:
    documents.append((r,"pos"))
for r in short_neg_sent:
    documents.append((r,"neg"))

all_words=[]
for i in range(len(short_pos_sent)):
##	print i
	words=(word_tokenize(short_pos_sent[i]))
	for word in words:
		all_words.append(word.lower())
del short_pos_sent

for i in range(len(short_neg_sent)):
##	print i
	words=(word_tokenize(short_neg_sent[i]))
	for word in words:
		all_words.append(word.lower())
del short_neg_sent

print"DATA donnnnnnnne"
##time.sleep(10)

save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()
##################################################################################################


###################  TOP 20000 words for tweets   ################################################


all_words_freq=nltk.FreqDist(all_words)
del all_words
word_features=all_words_freq.keys()[:5000]
del all_words_freq

save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


##################################################################################################



#################     FEATURE EXTRACTION FOR TWEETS     ###########################################
def find_features(doc_words):
    uniq_words=set(word_tokenize(doc_words))
    features={}
    for w in word_features:
        features[w]=(w in uniq_words)
    return features
###################################################################################################

######################NOW YOU HAVE FEATURE SET FOR EVERY REVIEW/TWEETS ############################
feature_set=[()]
for (word_set,categ) in documents:
    feature_dict={}
    
    feature_dict=find_features(word_set)
    
    feature_set.append((feature_dict,categ))
feature_set.append(({'not':True},'neg'))
random.shuffle(feature_set)

save_documents = open("pickled_algos/featuresets.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

###################################################################################################

###########################  training and testing set    ##########################################
training_set=feature_set[1000:2000]
testing_set=feature_set[10000:]
del feature_set
check=['pos','neg']
nos=[]
for i in range(len(training_set)):
	try:
		if(training_set[i][1] not in check):
			pass
	except:
		nos.append(i)
for i in nos:
    del training_set[i]
###################################################################################################


##############################       Classifiers ##################################################

acc={}
class_list={}

classifier=nltk.NaiveBayesClassifier.train(training_set)
class_list["NB"]=classifier
##classifier_file=open("my_first.pickle","r")
##classifier=pickle.load(classifier_file)
##classifier_file.close()

print("Original Showing accuracy"),
print(nltk.classify.accuracy(classifier,testing_set))
classifier.show_most_informative_features(10)
##print("votes:::",classifier.classify(testing_set[0][0]))
save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
del save_classifier
del classifier

##classifier_file=open("my_first.pickle","w")
##pickle.dump(classifier,classifier_file)
##classifier_file.close()
##print("file is saved")



MNB_classifier=SklearnClassifier(MultinomialNB())
class_list["MNB"]=MNB_classifier.train(training_set)
print("mnb done")
save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()
del save_classifier
del MNB_classifier


####GNB_classifier=SklearnClassifier(GaussianNB())
########class_list[1]=
####GNB_classifier.train(training_set)
####print(nltk.classify.accuracy(GNB_classifier,testing_set))

BNB_classifier=SklearnClassifier(BernoulliNB())
class_list["BNB"]=BNB_classifier.train(training_set)
print("BNB done")
save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BNB_classifier, save_classifier)
save_classifier.close()
del save_classifier
del BNB_classifier


LRG_classifier=SklearnClassifier(LogisticRegression())
class_list["LRG"]=LRG_classifier.train(training_set)
print("LRG done")
save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LRG_classifier, save_classifier)
save_classifier.close()
del save_classifier
del LRG_classifier


SGD_classifier=SklearnClassifier(SGDClassifier())
class_list["SGD"]=SGD_classifier.train(training_set)
print("SGD done")
save_classifier = open("pickled_algos/SGD_classifier5k.pickle","wb")
pickle.dump(SGD_classifier, save_classifier)
save_classifier.close()
del save_classifier
del SGD_classifier


##SVC_classifier=SklearnClassifier(SVC())
####class_list.append(SVC_classifier.train(training_set))
##class_list["SVC"]=SVC_classifier.train(training_set)

LSVC_classifier=SklearnClassifier(LinearSVC())
class_list["LSVC"]=LSVC_classifier.train(training_set)
print("LSVC done")
save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LSVC_classifier, save_classifier)
save_classifier.close()
del save_classifier
del LSVC_classifier

NSVC_classifier=SklearnClassifier(NuSVC())
class_list["NSVC"]=NSVC_classifier.train(training_set)
print("NSVC done")
save_classifier = open("pickled_algos/NUSVC_classifier5k.pickle","wb")
pickle.dump(NSVC_classifier, save_classifier)
save_classifier.close()
del save_classifier
del NSVC_classifier



for i in range(len(class_list)):
    acc[class_list.keys()[i]]=nltk.classify.accuracy(class_list.values()[i],testing_set)
    
##########################################################################################################

    

############################  CONFIDENCE MEASURE OF TESTING SET     ########################################
for i in range(len(testing_set)):
    print(documents[i][0])
    votes=[]
    for cl in class_list.values():
        votes.append(cl.classify(testing_set[i][0]))
    choice_votes = votes.count(mode(votes))
    conf = float(choice_votes) / len(votes)
    print("voted_classifier",mode(votes),"with confidence",conf)
##    conf=voted_classifier.confidence(testing_set[i][0])
##    print classifier.classify(testing_set[i][0])
##    print(v,"==",conf*100)

#############################################################################################################
