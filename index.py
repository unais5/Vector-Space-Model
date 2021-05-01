import nltk
import csv
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import OrderedDict
import operator
from tkinter import *

class Indexer:
    def __init__(self):
        self.tokens = []
        self.docToken = {}
        self.document_vector={}
        self.termfreq = {}
        self.idf = {}
        self.tfidf={}
        self.query_vector = {}
        self.stopWord = []
        self.cosine = {}
        
    def Preprocessing(self):
        #  Loading Data from files
        Doc={}
        for x in range(50):
            x=x+1
            file="ShortStories/"+str(x)+".txt"
            Doc[x]=open(file,'r',encoding='utf-8').read()
            
        #  Storing all the data at a single place
        allDoc=""
        for x in range(50):
            x=x+1
            allDoc=allDoc+" \n"+Doc[x]
        #  Importing Stopwords
        self.stopWord=open("files/Stopword-List.txt").read()
        
        #  Tokenizing the documents and stop words
        self.tokens=nltk.word_tokenize(allDoc)
        self.stopWord=nltk.word_tokenize(self.stopWord)
        
        #  Creating unique tokens
        self.tokens=list(set(self.tokens))
        
        #  Remove special characters
        removetable=str.maketrans("", "", "'!@#$%^&*()_=-\|][:';:,<.>/?`~")
        self.tokens=[x.translate(removetable) for x in self.tokens]
        
        #  Decapitalized
        self.tokens=[element.lower() for element in self.tokens]
        
        # Removing StopWords
        self.tokens=[x for x in self.tokens if x.isalnum() and x not in self.stopWord]
        self.tokens=[element.lower() for element in self.tokens]
        
        #  Sorting Tokens
        self.tokens=sorted(self.tokens)
        
        #  Document wise Tokenization
        for x in range(1,50):
            self.docToken[x]=nltk.word_tokenize(Doc[x])
            
        #  Remove Special characters documnet wise
        removetable=str.maketrans("", "", "'!@#$%^&*()_=-\|][:';:,<.>/?`~")
        for x in range(1,50):
            self.docToken[x]=[y.translate(removetable) for y in self.docToken[x]]

        #  Documnet wise sorting
        for x in range(1,50):
            self.docToken[x]=sorted(self.docToken[x])

        #  Decaptilized document wise
        for x in range(1,50):
            self.docToken[x]=[element.lower() for element in self.docToken[x]]

        for x in range(1,50):
            self.docToken[x]=[y for y in self.docToken[x] if y.isalnum() and y not in self.stopWord]
            
        #  Calling Term freq function
        self.Find_TermFreq()
        
        #  Calling idf function
        self.Find_idf()
        
        # Calling tfidf function
        self.Find_tfidf()
        
        
    def Find_TermFreq(self):
        #  Creating dictionary for each token and assign value zero
        for x in range(1,50):
            self.document_vector[x]=dict.fromkeys(self.tokens,0)
            
        #  Add frequncy for each term
        for x in range(1,50):
            for word in self.docToken[x]:
                self.document_vector[x][word]+=1 
                
        #  tf
        for x in range(1,50):
            self.termfreq[x]={}
            for word,count in self.document_vector[x].items():
                self.termfreq[x][word]=count
        #  Writing term frequencies in the file
        term_frequency_file=open('files/Term Frequency.txt','w')
        for elem in self.termfreq:
            term_frequency_file.write(str(elem) + ':\n' + str(self.termfreq[elem]) + '\n\n')
        term_frequency_file.close()
        
    def Find_idf(self):
        #  unique Token document wise
        for x in range(1,50):
            self.docToken[x]=set(self.docToken[x])
            self.docToken[x]=list(set(self.docToken[x]))

        wordDcount=dict.fromkeys(self.tokens,0)
        for word in self.tokens:
            for x in range(1,50):
                if word in self.docToken[x]:
                    wordDcount[word]+=1

        #  finding idf            
        for word in self.tokens:
            if wordDcount[word]>0:
                count=wordDcount[word]
                if count>50:
                    count=50
            self.idf[word]=math.log(50/count)
            
        #  Writing idf in the file
        idf_file=open('files/idf.txt','w')
        for elem in self.idf:
            idf_file.write(str(elem) + ':' + str(self.idf[elem]) + '\n\n')
        idf_file.close()
        
    def Find_tfidf(self):
        for x in range(1,50):
            self.tfidf[x]={}
            for word in self.document_vector[x]:
                self.tfidf[x][word]=self.termfreq[x][word]*self.idf[word]
                
        #  Writing tfidf in the file
        tfidf_file=open('files/tf idf.txt','w')
        for elem in self.tfidf:
            tfidf_file.write(str(elem) + ':' + str(self.tfidf[elem]) + '\n\n')
        tfidf_file.close()
        
    def Query_processing(self):
        #  Take query from frontend
        query=e1.get()
        #  Tokenize Query
        qt=nltk.word_tokenize(query)

        #  Remove Special characters
        removetable=str.maketrans("", "", "'!@#$%^&*()_=-\|][:';:,<.>/?`~")
        qt=[x.translate(removetable) for x in qt]

        #  Decapitalized
        qt=[element.lower() for element in qt]

        #  Removind stopwords and making unique
        qt=[y for y in qt if y.isalnum() and y not in self.stopWord]
        qt=list(set(qt))

        self.query_vector=dict.fromkeys(self.tokens,0)
        for word in qt:
            try:
                self.query_vector[word]+=1
            except KeyError:
                None

        #  Query idf
        for words in self.query_vector:
            try:
                self.query_vector[words]=self.query_vector[words]*self.idf[word]
            except KeyError:
                None
                
        self.cosine_sim()
        
        lst={}
        for items in self.cosine:
            if items[1]>=0.005:
                lst[items[0]]=items[1]

        lst1=list(lst.keys())

        lst2=list(lst.values())
        lst2=["%.5f" % v for v in lst2]

        e2.delete(0,END)
        e3.delete(0,END)
        e4.delete(0,END)

        e2.insert(15,lst1)
        e3.insert(15,lst2)
        e4.insert(15,len(self.cosine)-len(lst))

                
    def cosine_sim(self):
        temp=0
        vec1=np.array([list(self.query_vector.values())])
        for x in range(1,50):
            vec2=np.array([list(self.tfidf[x].values())])
            if cosine_similarity(vec1,vec2)>0:
                temp=cosine_similarity(vec1,vec2)[0][0]
                self.cosine[x]=temp

        self.cosine=sorted(self.cosine.items(), key=operator.itemgetter(1), reverse=True)

        
if __name__ == "__main__":
    indexer = Indexer()
    indexer.Preprocessing()

    master = Tk()
    master.geometry('1000x300')
    master.title("Vector Space Model")

    Label(master, text="Enter Query: ",width=20,font=("bold", 10),padx=10, pady=20).grid(row=0)
    Label(master, text="Result (Document IDs (1st,2nd,3rd,...))",width=30,font=("bold", 10),padx=10, pady=20).grid(row=1)
    Label(master, text="Tf-IDF Rank",width=20,font=("bold", 10),padx=10, pady=20).grid(row=2)
    Label(master, text="Document Trimed (alpha=0.005)",width=20,font=("bold", 10),padx=10, pady=20).grid(row=3)

    e1 = Entry(master,width=100)
    e2 = Entry(master,width=100)
    e3 = Entry(master,width=100)
    e4 = Entry(master,width=100)

    e1.insert(15,"crowd busy")

    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)
    e3.grid(row=2, column=1)
    e4.grid(row=3, column=1)

    Button(master, text='Search', command=indexer.Query_processing()).grid(row=5, column=1, sticky=W, pady=4)

    mainloop( )
