
'''
   utility functions for processing terms

    shared by both indexing and query processing
'''
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from norvig_spell import correction


def Tokenize(word):
    #ntlk function to tokenize the words
    removable_chars = [';', ':', '!', '*','.',',','/']
    word = ''.join(i for i in word if not i in removable_chars)
    return word_tokenize(word)

def spellcheck(word):
    spellcheckedwords=[]
    for x in word:
        wrd=correction(x)
        spellcheckedwords.append(wrd)
    return spellcheckedwords

def ConvertLowerCase(word):
    return word.lower()

def ReadStopWordsFile():
    stopwordslist=[]
    stopwordsFile = open("stopwords", "r")
    stopwords=stopwordsFile.read()
    stopwords=stopwords.split('\n')
    return stopwords

def isStopWord(word):
    ''' using the NLTK functions, return true/false'''
    #check with nltk stopwords as well even after removing from stopword files
    # ToDo
    stop_words = set(stopwords.words("english"))
    if word in stop_words:
        return True
    else:
        return False

def removeStopWords(words):
    stopwords=ReadStopWordsFile()
    stopwordsremoval=[]
    for x in words:
        if(x not in stopwords):
          if(not isStopWord(x)):
              stopwordsremoval.append(x)
    return stopwordsremoval





def stemming(word):
    ''' return the stem, using a NLTK stemmer. check the project description for installing and using it'''

    # ToDo
    porterstem = PorterStemmer()
    stemmedwords=[]
    for x in word:
        stemmedwords.append(porterstem.stem(x))
    return stemmedwords

if __name__ == '__main__':
    print(ReadStopWordsFile())