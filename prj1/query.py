
'''
query processing

'''
import math
import random
import sys
from time import process_time
import cran
from cranqry import loadCranQry
from index import InvertedIndex
from util import *
from functools import reduce
from heapq import nlargest

class QueryProcessor:

    def __init__(self, query, index, collection,query_nid):
        ''' index is the inverted index; collection is the document collection'''
        self.raw_query = query
        self.index=index
        self.docs = collection
        self.querynumber=query_nid
    def preprocessing(self):
        ''' apply the same preprocessing steps used by indexing,
            also use the provided spelling corrector. Note that
            spelling corrector should be applied before stopword
            removal and stemming (why?)'''
        # ToDo: return a list of terms
        querynumber="{0:0=3d}".format(self.querynumber)
        #print(querynumber)
        if(querynumber not in self.raw_query.keys()):
            print("Query Number does not exists")
            exit(0)
        inputquery=self.raw_query[querynumber].text
        #print(inputquery)
        #tokenize the query words
        lowered_documentWords=ConvertLowerCase(inputquery)
        tokenizedWords=Tokenize(lowered_documentWords)
        spellcheckedwords=spellcheck(tokenizedWords)
        removedstopwords=removeStopWords(spellcheckedwords)
        stemmedWords=stemming(removedstopwords)

        return stemmedWords


    def booleanQuery(self):
        ''' boolean query processing; note that a query like "A B C" is transformed to "A AND B AND C" for retrieving posting lists and merge them'''
        #ToDo: return a list of docIDs
        preproces_terms=self.preprocessing()
        #print(preproces_terms)
        listdocidterms=[]
        for term in preproces_terms:
            if term in self.index.items.keys():
             documentIDlist=list(self.index.items.get(term).get('posting').keys())
             listdocidterms.append(documentIDlist)
            else:
                listdocidterms.append([])
        #print(listdocidterms)
        res = list(reduce(lambda i, j: i & j, (set(x) for x in listdocidterms)))
        #print(res)
        return res


    def vectorQuery(self, k,alt=False):
        ''' vector query processing, using the cosine similarity. '''
        #ToDo: return top k pairs of (docID, similarity), ranked by their cosine similarity with the query in the descending order
        # You can use term frequency or TFIDF to construct the vectors
        #if alt false then weight are caluclted as (query)ltc.ltc(document) and alt is true then (query) apc.ltc (docuemnt)
        preproces_terms = self.preprocessing()
        documents=self.docs
        #calculate tf-idf of query words
        #we are gona caculate using tf-idf = termfequency * idf
        highest_term_fequency=0;
        if(alt):
            for x in preproces_terms:
                xls=preproces_terms.count(x)
                if(xls>highest_term_fequency):
                    highest_term_fequency=xls
        qcvector={}
        for t in preproces_terms:
            if(not alt):
             tf=preproces_terms.count(t)
            else:
              tfk = preproces_terms.count(t)
              tf=0.5+((0.5*tfk)/highest_term_fequency)
            if(t in self.index.items):
             if(not alt):
              tfidf=(1+ math.log(tf,10) ) * (self.index.items[t].get('idf')) #((self.index.items[query_tokens[temp2]].get('idf') )* (1 + math.log( wordfreq[0] , 10)))
             else:
                n=self.index.nDocs
                #print(self.index.items.get(t))
                #print(type(self.index.items.get(t)))
                df=len(list(self.index.items.get(t)['posting'].keys()))
                idf_k=math.log(((n-df)/df),10)
                if(0<idf_k):
                    tfidf=tf*idf_k
                else:
                    tfidf=0
            else:
              tfidf=0
            qcvector[t]=tfidf
        # caculate the tf-idf for each document words and the then caculate the cosine similarity between document and query
        finalvector={}

        for doc in self.docs.docs:
            dcvector = {}
            title = doc.title
            body = doc.body
            # First convert both title and body words and lowercase
            documentWords = title + body
            #lowercasewords=ConvertLowerCase(doc)
            #words=Tokenize(lowercasewords)
            lowered_documentWords = ConvertLowerCase(documentWords)
            tokenizedWords = Tokenize(lowered_documentWords)
            removedstopwords = removeStopWords(tokenizedWords)
            words = stemming(removedstopwords)
            for w in words:
                tf=1+ math.log(self.index.items[w].get('posting').get(doc.docID).get('term_frequency'),10)
                idf=self.index.items[w].get('idf')
                tf_idf=tf*idf
                dcvector[w]=tf_idf
            #print(dcvector)
            sqrt_sums_dc=0
            for x in dcvector:
                square=dcvector[x] * dcvector[x]
                sqrt_sums_dc+=square
            try:
             reciprocalsqrt_doc=(1/math.sqrt(sqrt_sums_dc))
            except:
                reciprocalsqrt_doc=0
            #caclulate d1/sqrt(d1^2) [sqrt(d1^2) is calculated above as reciprocalsqrt_doc]
            for x in dcvector:
                dcvector[x] *=reciprocalsqrt_doc
            #caculate cosine similarity between the Query words and docuemnt words
            cosinevector=qcvector.copy();
            i=0;
            while i < qcvector.__len__():
                qw=list(qcvector.keys())[i]
                if(qw in dcvector ):
                    cosinevector[qw]=dcvector[qw]*qcvector[qw]
                else:
                    cosinevector[qw]=0
                i=i+1
            #find the sum of cosine value
            sum_cosine_dc=0
            for x in cosinevector:
                sum_cosine_dc+=cosinevector[x]
            finalvector[doc.docID]=sum_cosine_dc
        #print(finalvector)
        getTop3Results=[]
        topthree = nlargest(k, finalvector, key=finalvector.get)
        for key in topthree:
            #print(key, ":", finalvector.get(key))
            getTop3Results.append((key,finalvector.get(key)))
        return getTop3Results

    def BatchEvaluation(self):
        TotalqueryTimeDic={}
        number_of_queries = self.querynumber
        docs = [int(x) for x in self.raw_query.keys()]
        for i in range(0,5):
           query_id_list=random.sample(docs,number_of_queries)
           print(query_id_list)
           #booleanVector for each query:
           booleanTime=0
           for query_id in query_id_list:
             self.querynumber=query_id
             start = process_time()
             res=self.booleanQuery()
             stop = process_time()
             booleanTime+=(stop-start)
           TotalqueryTimeDic[str(i+1)]={"booleanModel":booleanTime}
           vectorTime=0
           for query_id in query_id_list:
               self.querynumber=query_id
               start=process_time()
               res=self.vectorQuery(3)
               stop=process_time()
               vectorTime+=(stop-start)
           TotalqueryTimeDic[str(i+1)].update({'vectorModel':vectorTime})
        print(TotalqueryTimeDic)
        with open('Processing_time.csv', 'w') as f:
            f.write("%s,%s,%s\n" % ("Iteration", "booleanModel(seconds)", "vectorModel(seconds)"))
            for key in TotalqueryTimeDic.keys():
                f.write("%s,%s,%s\n" % (key, TotalqueryTimeDic[key].get('booleanModel'), TotalqueryTimeDic[key].get('vectorModel')))


def test(obj):
    ''' test your code thoroughly. put the testing cases here'''
    #Convert A sample Query to Terms

    sample_query="what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft ."
    print("Sample Query Considered:: ",sample_query)
    lowered_documentWords = ConvertLowerCase(sample_query)
    tokenizedWords = Tokenize(lowered_documentWords)
    spellcheckedwords = spellcheck(tokenizedWords)
    removedstopwords = removeStopWords(spellcheckedwords)
    stemmedWords = stemming(removedstopwords)
    print("Sample Query Converted to Terms::",stemmedWords)
    #run one boolean model for specific query and test the result
    obj.querynumber=29
    res=obj.booleanQuery()
    assert res == ['462']
    print("Test for Boolean Model Passed")

    #run one vector model for specific query and test the result
    res=obj.vectorQuery(3)
    vector_label = [x[0] for x in res]
    assert vector_label == ['462','1099','1340']
    print('Test for Vector Model Passed')

def query(index_file,model_type,query_file,query_id):
    ''' the main query processing program, using QueryProcessor'''

    # ToDo: the commandline usage: "echo query_string | python query.py index_file processing_algorithm"
    # processing_algorithm: 0 for booleanQuery and 1 for vectorQuery
    # for booleanQuery, the program will print the total number of documents and the list of docuement IDs
    # for vectorQuery, the program will output the top 3 most similar documents
    #load documents
    inputdocument = cran.CranFile("cran.all")
    #load the index file saved at from part 1
    index=InvertedIndex().load(index_file)
    #load query processed files
    queries=loadCranQry(query_file)

    qp=QueryProcessor(queries,index,inputdocument,query_id)

    if model_type==0:
        Booleanres=qp.booleanQuery()
        print(Booleanres)
    if model_type==1:
        vectorres=qp.vectorQuery(3)
        print(vectorres)
    if model_type==2:
        qp.BatchEvaluation()
    #print("********Running Test Cases ************")
    #test(qp)


if __name__ == '__main__':
    #test()
    #python index_file mode_selection query.text qid_or_n
    query(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))

