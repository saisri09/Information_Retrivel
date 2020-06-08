import scipy
from scipy import stats
import cran
import metrics
from batch_eval import process_querls_file
from cranqry import loadCranQry
from index import InvertedIndex
from query import *


def VectorCompare():
     queries = loadCranQry("query.text")
     queries_id_list=[str(int(x)) for x in queries.keys()]
     inputdocument = cran.CranFile("cran.all")
     # load the index file saved at from part 1
     index = InvertedIndex().load("index_file")
     qp = QueryProcessor(queries, index, inputdocument, 10)
     queries_id_list=[str(int(x)) for x in queries.keys()]
     #print(queries_id_list)
     #read querls.txt
     qrels_dict=process_querls_file("qrels.text",queries_id_list)
     #IdeaVectorsforQuery_ids={}
     sumbooleanNADC=[]
     sumvectorNADC=[]
     vectorNADC1 = []
     booleanNADC2 = []
     # random_query_id_list=[153, 18]
     # print(random_query_id_list)
     query_id = [4 , 29, 53, 58, 100]
     vectorNADC1=[]
     vectorNADC2=[]
     for q_id in query_id:
         qp.querynumber = q_id
         # boolean_res=qp.booleanQuery()
         vector_top3 = qp.vectorQuery(5)
         vector2_top3=qp.vectorQuery(5,True)
         # vector_top3=[('12',0.34),('746',0.33),('875',0.24)]
         # print(boolean_res)
         print("Output for Vector Model Result::", vector_top3)
         if (vector_top3.__len__() < 1):
             vectorNADC1.append(0)
         else:
             vector_label = [x[0] for x in vector_top3]
             score = [x[1] for x in vector_top3]
             print("DocumentIDs of Vector Model Result:: ", vector_label)
             print("Scores of Vector Model Result::", score)
             true_label = vector_label.copy()
             query_id = str(q_id)
             for x in vector_label:
                 # str_x="{0:0=3d}".format(x)
                 ind = vector_label.index(x)
                 if (x in qrels_dict.get(query_id)):
                     true_label[ind] = 1
                 else:
                     true_label[ind] = 0
             if true_label.__len__() < 5:
                 len_val = 10 - (true_label.__len__())
                 true_label.extend([0] * len_val)
             print("Actual Vector:: ", true_label)
             print("Predicted Vector:: ", score)
             if sum(true_label) == 0:
                 vectorNADC1.append(0)
             else:
                 ndcg = metrics.ndcg_score(true_label, score, 5)
                 print("Calculated ndcg for Vector::", ndcg)
                 vectorNADC1.append(ndcg)
         if (vector2_top3.__len__() < 1):
             vectorNADC2.append(0)
         else:
             vector_label = [x[0] for x in vector2_top3]
             score = [x[1] for x in vector2_top3]
             print("DocumentIDs of Vector Model Result:: ", vector_label)
             print("Scores of Vector Model Result::", score)
             true_label = vector_label.copy()
             query_id = str(q_id)
             for x in vector_label:
                 # str_x="{0:0=3d}".format(x)
                 ind = vector_label.index(x)
                 if (x in qrels_dict.get(query_id)):
                     true_label[ind] = 1
                 else:
                     true_label[ind] = 0
             if true_label.__len__() < 5:
                 len_val = 10 - (true_label.__len__())
                 true_label.extend([0] * len_val)
             print("Actual Vector:: ", true_label)
             print("Predicted Vector:: ", score)
             if sum(true_label) == 0:
                 vectorNADC2.append(0)
             else:
                 ndcg = metrics.ndcg_score(true_label, score, 5)
                 print("Calculated ndcg for Vector::", ndcg)
                 vectorNADC2.append(ndcg)
     print("Calculated NADC sum for all queries", vectorNADC1)
     avergae_vectorNADC = float(sum(vectorNADC1) / 5)
     print("Calculated NADC sum for all queries", vectorNADC2)
     avergae_vectorNADC2 = float(sum(vectorNADC2) / 5)
     print("Avergae NADC Vector::", avergae_vectorNADC)
     print("Avergae NADC boolean::", avergae_vectorNADC2)
     print(vectorNADC1)
     print(vectorNADC2)
     p_value = scipy.stats.wilcoxon(vectorNADC1, vectorNADC2, zero_method='wilcox', correction=False)
     p = "%.20f" % float(str(p_value[1]))
     print('P value for all the queries processed is:', p)

if __name__ == '__main__':
    VectorCompare()