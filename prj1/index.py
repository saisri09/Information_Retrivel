

'''

Index structure:

    The Index class contains a list of IndexItems, stored in a dictionary type for easier access

    each IndexItem contains the term and a set of PostingItems

    each PostingItem contains a document ID and a list of positions that the term occurs

'''
import math
import pickle
import sys
import jsonpickle
from util import *
import doc
import cran

class Posting:
    def __init__(self, docID):
        self.docID = docID
        self.positions = []
        self.term_frequency=0

    def append(self, pos):
        self.positions.append(pos)
        self.term_frequency=len(pos)

    def sort(self):
        ''' sort positions'''
        self.positions.sort()

    def merge(self, positions):
        self.positions.extend(positions)

    def term_freq(self):
        ''' return the term frequency in the document'''
        #ToDo


class IndexItem:
    def __init__(self, term):
        self.term = term
        self.posting = {} #postings are stored in a python dict for easier index building
        #self.sorted_postings= [] # may sort them by docID for easier query processing
        #self.termFrequency=0
    def add(self, docid, pos):
        ''' add a posting'''
        if docid not in self.posting.keys():
            self.posting[docid] = Posting(docid)
        self.posting[docid].append(pos)

    def sort(self):
        ''' sort by document ID for more efficient merging. For each document also sort the positions'''
        # ToDo


class InvertedIndex:

    def __init__(self):
        self.items = {} # list of IndexItems
        self.nDocs = 0  # the number of indexed documents

    def Find_positions(self,docwordlist,word):
        words_pos=[]
        position=0
        for i in range(len(docwordlist)):
            if docwordlist[i] == word:
                    words_pos.append(i+1)
        return words_pos


    def indexDoc(self, doc): # indexing a Document object
        ''' indexing a docuemnt, using the simple SPIMI algorithm, but no need to store blocks due to the small collection we are handling. Using save/load the whole index instead'''

        # ToDo: indexing only title and body; use some functions defined in util.py
        # Consider title and body of Docuemnts for indexing
        # (1) convert to lower cases,
        # (2) Tokenizing
        # (3) remove stopwords,
        # (4) stemming
        title=doc.title
        body=doc.body
        #First convert both title and body words and lowercase
        documentWords=title+body
        lowered_documentWords=ConvertLowerCase(documentWords)
        tokenizedWords=Tokenize(lowered_documentWords)
        removedstopwords=removeStopWords(tokenizedWords)
        stemmedWords=stemming(removedstopwords)
        #print(stemmedWords)
        self.nDocs=self.nDocs+1
        documentWord={}
        # consider the stemmed words for indexing
        for x in stemmedWords:
           if(x not in documentWord.keys()):
            doc_pos= self.Find_positions(stemmedWords, x)
            if(x not in self.items.keys()):
             indexitemobj=IndexItem(x)
             indexitemobj.add(self.nDocs,doc_pos)
             self.items[x]=indexitemobj
            else:
                indexitemobj = IndexItem(x)
                indexitemobj.add(self.nDocs, doc_pos)
                self.items.get(x).add(self.nDocs, doc_pos)
           documentWord[x]=doc_pos
        #print(documentWord)
        #self.updateIndex(documentWord)

    def sort(self):
        ''' sort all posting lists by docID'''
        #ToDo
        #sort is being inbuilt based on positions as we use Find_positions method

    def find(self, term):
        return self.items[term]

    def save(self, filename):
        ''' save to disk'''
        # ToDo: using your preferred method to serialize/deserialize the index
        print("Number of terms being Saved to Index_File",self.items.__len__())
        jsonEncoded = jsonpickle.encode(self)
        f=open(filename,'w')
        f.write(jsonEncoded)
        print("Index File Generated as ",filename)

    def load(self, filename):
        ''' load from disk'''
        # ToDo
        filepointer=open(filename,'r')
        indexRead=jsonpickle.decode(filepointer.read())
        return indexRead
    def idf(self, term):
        ''' compute the inverted document frequency for a given term'''
        #ToDo: return the IDF of the term
        idf_value = self.nDocs/len(self.items[term].posting.keys())
        self.items[term].idf = math.log(idf_value,10)

    # more methods if needed


def test(obj):
    ''' test your code thoroughly. put the testing cases here'''
    #StopWords Removal,lowercase,tokenize,stemming.
    document="experimental investigation of the aerodynamics of a wing in a slipstream "
    print("Sample Document considered ::",document)
    lowered_documentWords = ConvertLowerCase(document)
    tokenizedWords = Tokenize(lowered_documentWords)
    removedstopwords = removeStopWords(tokenizedWords)
    stemmedWords = stemming(removedstopwords)
    stopwords_Removed=' '.join(removedstopwords)
    print("Sample Document After Removal of StopWords ::",stopwords_Removed)
    print("Terms After stemmed ::",stemmedWords)
    #loading Index File
    IndexobjectRead=obj.load("index_file")
    print('Loading Index File Test Case Passed')


def indexingCranfield(data_file,indexfile):
    #ToDo: indexing the Cranfield dataset and save the index to a file
    # command line usage: "python index.py cran.all index_file"
    # the index is saved to index_file
    #first load the document file
    inputdocument= cran.CranFile(data_file)

    #create object for invetedIndex class
    #iterate over the document File and create the index file
    #calculate idf
    invertedobj=InvertedIndex()
    for docs in inputdocument.docs:
         invertedobj.indexDoc(docs)
    for x in invertedobj.items:
         invertedobj.idf(x)
    invertedobj.save(indexfile)


if __name__ == '__main__':
    #python index.py cran.all index_file
    indexingCranfield('cran.all','index_file')
    #indexingCranfield(str(sys.argv[1]), str(sys.argv[2]))

