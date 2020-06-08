import datetime

from matplotlib import pyplot
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

multinominal_chi=[]
multinominal_mutal=[]
bernouli_chi=[]
bernouli_mutual=[]
knn_chi=[]
knn_mutual=[]
c_support_chi=[]
c_support_mutal=[]
#print(datetime.datetime.now())
#kvals=[500, 1500, 2500, 3500, 4500, 5500, 6500, 8500,10000]
print("Note:: This Programme run more than 20 minutes...")
#considered K values ranging from 500 to 2k
kvals=[*range(500,2000,200)]
print(kvals)
#iterate for all the values of k
for k_value in kvals:
  print("Processing feature_Selection on all algorithms for k-value:",k_value)
  clf = MultinomialNB()
  #load training file TF
  X, y = load_svmlight_file("training_data_file.TF")
  #select K best feature using ch-square method
  X_new1 = SelectKBest(chi2, k=k_value).fit_transform(X, y)
  scores = cross_val_score(clf, X_new1, y, cv=5, scoring='f1_macro')
  multinominal_chi.append(scores.mean())
  # select K best feature using mutualinfo method
  X_new2 = SelectKBest(mutual_info_classif, k=k_value).fit_transform(X, y)
  scores = cross_val_score(clf, X_new2, y, cv=5, scoring='f1_macro')
  multinominal_mutal.append(scores.mean())

  clf = BernoulliNB()
  X, y = load_svmlight_file("training_data_file.IDF")
  # select K best feature using ch-square method
  X_new1 = SelectKBest(chi2, k=k_value).fit_transform(X, y)
  scores = cross_val_score(clf, X_new1, y, cv=5, scoring='f1_macro')
  bernouli_chi.append(scores.mean())
  # select K best feature using mutualinfo method
  X_new2 = SelectKBest(mutual_info_classif, k=k_value).fit_transform(X, y)
  scores = cross_val_score(clf, X_new2, y, cv=5, scoring='f1_macro')
  bernouli_mutual.append(scores.mean())

  clf = KNeighborsClassifier()
  X, y = load_svmlight_file("training_data_file.TFIDF")
  # select K best feature using ch-square method
  X_new1 = SelectKBest(chi2, k=k_value).fit_transform(X, y)
  scores = cross_val_score(clf, X_new1, y, cv=5, scoring='f1_macro')
  knn_chi.append(scores.mean())
  # select K best feature using mutualinfo method
  X_new2 = SelectKBest(mutual_info_classif, k=k_value).fit_transform(X, y)
  scores = cross_val_score(clf, X_new2, y, cv=5, scoring='f1_macro')
  knn_mutual.append(scores.mean())

  clf = SVC()
  X, y = load_svmlight_file("training_data_file.TFIDF")
  # select K best feature using ch-square method
  X_new1 = SelectKBest(chi2, k=k_value).fit_transform(X, y)
  #apply cross val score method with scoring of f1_macro
  scores = cross_val_score(clf, X_new1, y, cv=5, scoring='f1_macro')
  c_support_chi.append(scores.mean())
  # select K best feature using mutualinfo method
  X_new2 = SelectKBest(mutual_info_classif, k=k_value).fit_transform(X, y)
  scores = cross_val_score(clf, X_new2, y, cv=5, scoring='f1_macro')
  c_support_mutal.append(scores.mean())


#plot figure for Chi-square
pyplot.figure(1)
#pyplot.subplot(211)
pyplot.plot(kvals, multinominal_chi,label = "Multinomial Naive Bayes")
pyplot.plot(kvals, bernouli_chi, label = "Bernoulli Naive Bayes")
pyplot.plot(kvals, knn_chi, label = "KNN")
pyplot.plot(kvals, c_support_chi, label = "SVM")
pyplot.xlabel("K")
pyplot.ylabel("f1_macro")
pyplot.title("CHI Square")
pyplot.legend(loc = 'best')
#plot figure for mutualinformation
pyplot.figure(2)
pyplot.plot(kvals, multinominal_mutal,label = "Multinomial Naive Bayes")
pyplot.plot(kvals, bernouli_mutual, label = "Bernoulli Naive Bayes")
pyplot.plot(kvals, knn_mutual, label = "KNN")
pyplot.plot(kvals, c_support_mutal, label = "SVM")
pyplot.xlabel("K")
pyplot.ylabel("f1_macro")
pyplot.title("Mutual Information")
pyplot.legend(loc = 'best')
#print(datetime.datetime.now())
pyplot.show()
