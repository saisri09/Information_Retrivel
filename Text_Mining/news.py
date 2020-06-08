import os
from os import walk
from doc import Document

class read_news():
    def __init__(self,newsdir):
        self.docs=[]
        filespaths = []
        #get path of newsdirectory
        #pathfile=os.getcwd() + "\\" + newsdir #"mini_newsgroups"
        pathfile=newsdir
        #loop in the subdirectoy and read the files
        for (dirpath, dirnames, filenames) in walk(pathfile):
             for x in dirnames:
               for (subdirpath, subdirnames, files) in walk(pathfile+"\\"+x):
                  for f in files:
                      self.readfiles(subdirpath,f,x)
    def readfiles(self,dirname,filename,subdir):
        #read file subject and last xx lines
        filepath=dirname+"\\"+filename
        cf = open(filepath)
        docid = filename+"_"+subdir
        number_of_lines=0
        title = ''
        body = ''
        linemessage = ''
        startlines=False
        for line in cf:
            if 'Subject:' in line:
                title = line[9:].strip()  # got title
            elif 'Lines:' in line:
                try:
                 number_of_lines=int(line[6:])
                except Exception as e:
                    if 'dog' in str(e):
                        number_of_lines=24
                startlines = True
                line = ''
            if startlines:
                #last_line = cf.readlines()[-number_of_lines:]
                last_line=[i.replace('\n','') for i in cf.readlines()[-number_of_lines:]]
                linemessage= ''.join(last_line)
        body = linemessage;
        #convert file to document format
        self.docs.append(Document(docid, title,body))
