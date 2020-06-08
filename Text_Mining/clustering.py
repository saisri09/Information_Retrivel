from matplotlib import pyplot
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

feature_vectors, targets = load_svmlight_file("training_data_file.TFIDF")
print("Note:: This Programme runs for approximatley 5 minutes...")
#K clsuters range from 2 to 25
clust_list=[*range(2,26)]
print(clust_list)
silhouette_score_kmeans=[]
normalized_score_kmeans=[]
silhouette_score_agglormative=[]
normalized_score_agglormative=[]

#consider 100 best features for clustering
print("Selecting 100 best features.....")
X=SelectKBest(mutual_info_classif, k=100).fit_transform(feature_vectors, targets).toarray()
#run for each cluster number
for n_clust in clust_list:
  print("Running for clusters:",n_clust)
  #apply kmeans clustering algorithm
  kmeans_model = KMeans(n_clusters=n_clust).fit(X)
  clustering_labels = kmeans_model.labels_
  #calculate sc score
  silhoutescore=metrics.silhouette_score(X, clustering_labels, metric='euclidean')
  silhouette_score_kmeans.append(silhoutescore)
  #calculate NMI score
  normalized_scores=metrics.normalized_mutual_info_score(targets, clustering_labels)
  normalized_score_kmeans.append(normalized_scores)

for n_clust in clust_list:
    #apply hierarchial clustering algorithm
    single_linkage_model = AgglomerativeClustering(n_clusters=n_clust, linkage='ward').fit(X)
    clustering_labels=single_linkage_model.labels_
    #calculate sc score
    silhoutescore = metrics.silhouette_score(X, clustering_labels, metric='euclidean')
    silhouette_score_agglormative.append(silhoutescore)
    #calculate NMI score
    normalized_scores = metrics.normalized_mutual_info_score(targets, clustering_labels)
    normalized_score_agglormative.append(normalized_scores)

#plot figue for sc
plot1=pyplot.figure(1)
pyplot.plot(clust_list, silhouette_score_kmeans,label="KMeans")
pyplot.plot(clust_list, silhouette_score_agglormative,label="Hierarchical clustering")
pyplot.title("Sihouette Coefficient Scores")
pyplot.xlabel("Number of Clusters")
pyplot.ylabel("The Measures")
pyplot.legend(loc='best')
#plot figure for NMI
plot2=pyplot.figure(2)
pyplot.plot(clust_list, normalized_score_kmeans,label="KMeans")
pyplot.plot(clust_list, silhouette_score_agglormative,label="Hierarchical clustering")
pyplot.title("Normalized Mutual Information Scores")
pyplot.xlabel("Number of Clusters")
pyplot.ylabel("The Measures")
pyplot.legend(loc='best')
pyplot.show()
