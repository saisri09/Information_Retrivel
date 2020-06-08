import warnings
import datetime

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
#run for multinomial navie Bayes classifier
print("*******multinomial Naive Bayes classifier********")
clf = MultinomialNB()
#load Term frquency file as features and targets for multinomial
feature_vectors, targets = load_svmlight_file("training_data_file.TF")
#run the cross_validation method  with f1_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("f1_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#run the cross_validation method  with precision_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='precision_macro')
print("precision_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#run the cross_validation method  with recall_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='recall_macro')
print("recall_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("\n")
#run for Naive Bayes classifier
print("********Naive Bayes classifier****************")
#load Term frequency file as features and targets for navie bayes
clf = BernoulliNB()
feature_vectors, targets = load_svmlight_file("training_data_file.IDF")
#run the cross_validation method  with f1_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("f1_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#run the cross_validation method  with precision_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='precision_macro')
print("precision_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#run the cross_validation method  with recall_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='recall_macro')
print("recall_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("\n")
#run for k-nearest neighbors classifier
print("***************k-nearest neighbors classifier ***********************")
clf = KNeighborsClassifier()
#load Term frequency file as features and targets for k-nn
feature_vectors, targets = load_svmlight_file("training_data_file.TFIDF")
#run the cross_validation method  with f1_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("f1_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#run the cross_validation method  with precision_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='precision_macro')
print("precision_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#run the cross_validation method  with recall_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='recall_macro')
print("recall_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("\n")
#run for C-Support Vector Classifier
print("****************C-Support Vector Classifier*****************")
clf = SVC()
#load Term frequency file as features and targets for svm bayes
feature_vectors, targets = load_svmlight_file("training_data_file.TFIDF")
#run the cross_validation method  with f1_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("f1_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#run the cross_validation method  with precision_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='precision_macro')
print("precision_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#run the cross_validation method  with recall_macro scoring and get the scores
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='recall_macro')
print("recall_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
