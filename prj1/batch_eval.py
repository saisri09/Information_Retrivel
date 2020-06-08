'''
a program for evaluating the quality of search algorithms using the vector model

it runs over all queries in query.text and get the top 10 results,
and then qrels.text is used to compute the NDCG metric

usage:
    python batch_eval.py index_file query.text qrels.text n

    output is the average NDCG over all the queries for boolean model and vector model respectively.
	also compute the p-value of the two ranking results. 
'''
#import re
import scipy
from scipy import stats
import cran
import metrics
from cranqry import loadCranQry
from index import InvertedIndex
from query import *


def process_querls_file(qrels_file,queries_id_list):
    qrels_dic={}
    f=open(qrels_file,'r').read()
    row=f.split("\n")
    #print(row)
    for x in row:
        if(x==''):
            break
        record=x.split(' ')
        quer_id=record[0]
        doc_id=record[1]
        if(quer_id in qrels_dic.keys()):
         qrels_dic[quer_id].append(doc_id)
        else:
         qrels_dic[quer_id]=[doc_id]
    #print(qrels_dic)
    #ignore the query_ids from querls.text which are not in query.text
    #print(queries_id_list)
    #print(qrels_dic)
    #print(len(qrels_dic.keys()))
    #print(queries_id_list.__len__())
    temp_dict = dict()
    k=0
    for x in qrels_dic.keys():
        newkey=queries_id_list[k]
        temp_dict[newkey] = qrels_dic[x]
        k+=1
        #temp_dict=dict(qrels_dic)
    # for x in temp_dict.keys():
    #     if x not in queries_id_list:
    #         del qrels_dic[x]
    #print(temp_dict)
    return temp_dict

def eval(index_file,query_file,qrels_File,number_of_queries):
    #read queryfile,indexfile
    # ToDo
    queries = loadCranQry(query_file)
    queries_id_list=[str(int(x)) for x in queries.keys()]
    #print(queries_id_list)
    #read querls.txt
    qrels_dict=process_querls_file(qrels_File,queries_id_list)
    inputdocument = cran.CranFile("cran.all")
    # load the index file saved at from part 1
    index = InvertedIndex().load(index_file)
    qp = QueryProcessor(queries, index, inputdocument, number_of_queries)
    queries_id_list_int=[int(x) for x in qrels_dict.keys()]
    queries_id_ls = [int(x) for x in queries.keys()]
    #IdeaVectorsforQuery_ids={}
    sumbooleanNADC=[]
    sumvectorNADC=[]
    with open('Evaluation_search.csv', 'w') as f:
     f.write("%s,%s,%s,%s\n" % ("Iteration", "AverageNDCG-booleanModel", "AverageNDCG-vectorModel","P-value"))
     for i in range(0,5):
        vectorNADC=[]
        booleanNADC=[]
        intersection_queries=list(set(queries_id_list_int) & set(queries_id_ls))
        random_query_id_list = random.sample(queries_id_list_int, number_of_queries)
        #random_query_id_list=[153, 18]
        #print(random_query_id_list)
        for q_id in random_query_id_list:
            print("Processing for Query ID ::",q_id)
            qp.querynumber=q_id
            #boolean_res=qp.booleanQuery()
            vector_top3=qp.vectorQuery(5)
            #vector_top3=[('12',0.34),('746',0.33),('875',0.24)]
            #print(boolean_res)
            print("Output for Vector Model Result::",vector_top3)
            if(vector_top3.__len__()<1):
                vectorNADC.append(0)
            else:
               vector_label=[x[0] for x in vector_top3]
               score=[x[1] for x in vector_top3]
               print("DocumentIDs of Vector Model Result:: ",vector_label)
               print("Scores of Vector Model Result::",score)
               true_label=vector_label.copy()
               query_id=str(q_id)
               for x in vector_label:
                 #str_x="{0:0=3d}".format(x)
                 ind=vector_label.index(x)
                 if (x in qrels_dict.get(query_id)):
                      true_label[ind]=1
                 else:
                     true_label[ind]=0
               if true_label.__len__()<5:
                  len_val=10-(true_label.__len__())
                  true_label.extend([0]*len_val)
               print("Actual Vector:: ",true_label)
               print("Predicted Vector:: ",score)
               if sum(true_label)==0 :
                 vectorNADC.append(0)
               else:
                 ndcg=metrics.ndcg_score(true_label, score,5)
                 print("Calculated ndcg for Vector::",ndcg)
                 vectorNADC.append(ndcg)
            boolean_res = qp.booleanQuery()
            print("output of boolean_res:: ",boolean_res)
            if boolean_res.__len__()<1:
                 booleanNADC.append(0)
            else:
                 score=[1]*len(boolean_res)
                 if(score.__len__()<5):
                     leng=5-(score.__len__())
                     score.extend([0]*leng)
                 true_label = boolean_res.copy()
                 query_id = str(q_id)
                 for x in boolean_res:
                     ind = boolean_res.index(x)
                     if (x in qrels_dict.get(query_id)):
                         true_label[ind] = 1
                     else:
                         true_label[ind] = 0
                 if true_label.__len__() < 5:
                     len_val = 10 - (true_label.__len__())
                     true_label.extend([0] * len_val)
                 print("Actual boolean:: ", true_label)
                 print("Predicted boolean:: ", score)
                 if sum(true_label) == 0:
                     booleanNADC.append(0)
                 else:
                     ndcg = metrics.ndcg_score(true_label, score, 5)
                     print("Calculated ndcg for Boolean::", ndcg)
                     booleanNADC.append(ndcg)
        print("Calculated NADC sum for all queries",vectorNADC)
        avergae_vectorNADC=float(sum(vectorNADC)/number_of_queries)
        print("Calculated NADC sum for all queries",booleanNADC)
        avergae_booleanNADC=float(sum(booleanNADC)/number_of_queries)
        print("Avergae NADC Vector::",avergae_vectorNADC)
        print("Avergae NADC boolean::",avergae_booleanNADC)
        p_value=scipy.stats.wilcoxon(vectorNADC, booleanNADC, zero_method='wilcox', correction=False)
        print(i,str(avergae_booleanNADC),str(avergae_vectorNADC),str(p_value[1]))
        p="%.20f" % float(str(p_value[1]))
        print('P value for all the queries processed is:',p)
        f.write("%s,%s,%s,%s\n" % (i+1, str(avergae_booleanNADC), str(avergae_vectorNADC),str(p)))
    print('Done')

if __name__ == '__main__':
    #eval('index_file', 'query.text', 'qrels.text', 50)
    eval(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4]))
    #eval()
