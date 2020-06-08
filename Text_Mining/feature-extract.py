import math
import os
import sys

import news
import time
import datetime
import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English


class Index:
    """ Inverted index datastructure """

    def __init__(self, tokenizer, stemmer=None, stopwords=None):
        """
        tokenizer   -- NLTK compatible tokenizer function
        stemmer     -- NLTK compatible stemmer
        stopwords   -- list of ignored words
        """
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.index = defaultdict(list)
        self.documents = {}
        self.unique_id = 0
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)

    def lookup(self, word):
        """
        Lookup a word in the index
        """
        word = word.lower()
        if self.stemmer:
            word = self.stemmer.stem(word)

        return [self.documents.get(id, None) for id in self.index.get(word)]

    def add(self, document):
        """
        Add a document string to the index
        """
        content = document.title+ document.body
        for token in [t.lower() for t in nltk.word_tokenize(content)]:
            if token in self.stopwords:
                continue

            if self.stemmer:
                token = self.stemmer.stem(token)

            if self.unique_id not in self.index[token]:
                self.index[token].append(document.docID)

        self.documents[self.unique_id] = document
        self.unique_id += 1

class feature_extract():

   def __init__(self):
       self.featureLookup={}
       self.class_map_dic={}
       self.index = Index(nltk.word_tokenize,
                     EnglishStemmer(),
                     nltk.corpus.stopwords.words('english'))
   def find_class(self,x):
       x=x.split("_",1)[1]
       for keys,values in self.class_map_dic.items():
           if x in values:
               return keys

   def calculate_idf(self,termfreq):
       idf_value = self.index.unique_id / termfreq
       idf = abs(math.log(idf_value, 10))
       return idf
   def remove_dupes(self,orglist):
       duplist=[]
       for x in orglist:
           if x not in duplist:
               duplist.append(x)
       return duplist
   def load_training_data_file_TF(self,training_file):
       #f=open(training_file.type,"w")
       tfdoclist={}
       idfdoclist={}
       tfidfdoclist={}
       for term in self.index.index.keys():
         for doc in self.remove_dupes(self.index.index[term]):
             termfreq=self.index.index[term].count(doc)
             idfval=self.calculate_idf(termfreq)
             tfidf=termfreq*idfval
             term_id=self.featureLookup[term]
             if(doc in tfdoclist.keys()):
                 tfdoclist.get(doc)[term_id]=termfreq
             else:
                 tfdoclist[doc]={term_id:termfreq}
             if(doc in idfdoclist.keys()):
                 idfdoclist.get(doc)[term_id]=idfval
             else:
                 idfdoclist[doc]={term_id:idfval}
             if(doc in tfidfdoclist.keys()):
                 tfidfdoclist.get(doc)[term_id]=tfidf
             else:
                 tfidfdoclist[doc]={term_id:tfidf}
       #write termfrequency file
       print("Loading Term Frequency Training data file..")
       f=open(training_file+".TF","w")
       for key,value in tfdoclist.items():
           docstring=self.find_class(key)
           docstring+=' '
           docstring+=str(value).replace(',','').replace(': ',':').split('{')[1].split('}')[0]
           docstring+='\n'
           f.write(docstring)
       f.close()
       print("Succesfully Loaded Term Frequency Training data file")
       print("Loading IDF Training data file.....")
       #write IDF file
       f=open(training_file+".IDF","w")
       for key,value in idfdoclist.items():
           docstring=self.find_class(key)
           docstring+=' '
           docstring+=str(value).replace(',',' ').replace(': ',':').split('{')[1].split('}')[0]
           docstring+='\n'
           f.write(docstring)
       f.close()
       print("Succesfully IDF Training data file")
       print("Loading TF-IDF Training data file.....")
       #write TFIDF file
       f=open(training_file+".TFIDF","w")
       for key,value in tfidfdoclist.items():
           docstring=self.find_class(key)
           docstring+=' '
           docstring+=str(value).replace(',',' ').replace(': ',':').split('{')[1].split('}')[0]
           docstring+='\n'
           f.write(docstring)
       f.close()
       print("Succesfully TF-IDF Training data file")
   def load_feature_definition_file(self,feature_file):
       f=open(feature_file,'w')
       #define the feature_id & initiate to 1
       ftr_id=1
       for trm in self.index.index.keys():
           f.write('('+str(ftr_id)+','+trm+')\n')
           self.featureLookup[trm] = ftr_id
           ftr_id=ftr_id+1
       f.close()

   def load_class_definition_file(self,class_file):
       f = open(class_file, "w")
       #declare the class mapping
       self.class_map_dic = {
           '1': ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                 'comp.windows.x'], '2': ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
           '3': ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'], '4': ['misc.forsale'],
           '5': ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
           '6': ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']}
       for key, value in self.class_map_dic.items():
           for x in value:
               f.write('(' + x + ',' + key + ')\n')
       f.close()

   def feature_extraction(self,newsdir,feature_file,class_file,training_file):
       #Load newdirectory files
       inputdocument= news.read_news(newsdir)
       print("Generating Index for documents (it might take approximately 57 seconds)....")
       #Perform the Indexing
       for doc in inputdocument.docs:
         self.index.add(doc)
       #Load class definition,feature definition,training files
       print("Loading Class definition file..")
       self.load_class_definition_file(class_file)
       print("Loading feature definition file")
       self.load_feature_definition_file(feature_file)
       print("Loading Training File")
       self.load_training_data_file_TF(training_file)

def test():
    #check whether read all files from directory given
    newsdoc=news.read_news("mini_newsgroups")
    assert len(newsdoc.docs) == 2000
    print("Test Case :: Loading newsdirectory-2k Documents PASSED")
    #cehck whether Index created after stop words removed and done stemmed for a document
    doc=newsdoc.docs[1]
    print("******* Document considered for Index testing::")
    print(doc.title+ doc.body)
    index = Index(nltk.word_tokenize,
                  EnglishStemmer(),
                  nltk.corpus.stopwords.words('english'))
    index.add(doc)
    indexstr=''
    for x in index.index.keys():
        indexstr+=x+' '
    print('***** Document after removal of stopwords and stemming *******')
    print(indexstr)
    print("Test Case :: Index created passed")
    #check whether feature_definition_file,class_definition_file,training_data_file created.
    if(os.path.exists('feature_definition_file')):
        print('Test Case :: Loading feature_definition_file passed')
    if(os.path.exists('class_definition_file')):
        print('Test Case :: Loading class_definition_file passed')
    from sklearn.datasets import load_svmlight_file
    feature_vectors, targets = load_svmlight_file("training_data_file.TF")
    print("Test Case :: Loading training_data_file.TF passed")
    from sklearn.datasets import load_svmlight_file
    feature_vectors, targets = load_svmlight_file("training_data_file.IDF")
    print("Test Case :: Loading training_data_file.IDF passed")
    from sklearn.datasets import load_svmlight_file
    feature_vectors, targets = load_svmlight_file("training_data_file.TFIDF")
    print("Test Case :: Loading training_data_file.TFIDF passed")

if __name__ == '__main__':
    feature_obj=feature_extract()
    feature_obj.feature_extraction("D:\\SAISRI\\IR\\Text_Mining\\mini_newsgroups","feature_definition_file","class_definition_file","training_data_file")
    #feature_obj.feature_extraction(str(sys.argv[1]), str(sys.argv[2]),str(sys.argv[3]),str(sys.argv[4]))