#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:11:20 2019

@author: samrat
"""

import pandas as pd
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Other
import re
import string
import numpy as np
import pandas as pd
import math


data=pd.ExcelFile('class2-9.xlsx')

df101=pd.read_excel(data,'101')
df104=pd.read_excel(data,'104')
df501=pd.read_excel(data,'501')
df701=pd.read_excel(data,'701')
df801=pd.read_excel(data,'801')

df101=df101['DESC']
df104=df104['Material Short Description']
df501=df501['Material Short Description']
df701=df701['Material Short Description']
df801=df801['Material Description']

#creating the initial dataframe to be used
df=pd.DataFrame(columns=['DESC','CLASS'])
c=0
d=0
for item in df101:
    if(item!="NOT TO BE USED"):
        df.loc[c]=[item,'101']
        c+=1
    else:
        d+=1
for item in df104:
    if(item!="NOT TO BE USED"):
        df.loc[c]=[item,'104']
        c+=1
    else:
        d+=1
for item in df501:
    if(item!="NOT TO BE USED"):
        df.loc[c]=[item,'501']
        c+=1
    else:
        d+=1
for item in df701:
    if(item!="NOT TO BE USED"):
        df.loc[c]=[item,'701']
        c+=1
    else:
        d+=1
for item in df801:
    if(item!="NOT TO BE USED"):
        df.loc[c]=[item,'801']
        c+=1
    else:
        d+=1
    
#cleaning text

df = df.dropna() # remove missing values
df=df.drop_duplicates()
df = df[df.DESC.apply(lambda x: x !="")]
token_count=0
def clean_text(text):
    global token_count
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    #stops = set(stopwords.words("english"))
    #text = [w for w in text if not w in stops and len(w) >= 2]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)#convert multiple whitespaces into single whitespace
    
    text = text.split()
    token_count+=len(text)
    #stemmer = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    #stemmed_words = [stemmer.stem(word) for word in text]
    #text = " ".join(stemmed_words)
    lemmatized_words = [wordnet.lemmatize(word, pos='v') for word in text]
    text = " ".join(lemmatized_words)

    return text

df['DESC'] = df['DESC'].map(lambda x: clean_text(x))
avg_words_per_sentence=math.ceil(token_count/len(df))

#removing any duplicates left

df=(df.reset_index()).drop('index',axis=1)
n=len(df)
remove_indexes=[]
for x in range(n):
    for y in range(x+1,n):
        if df['DESC'].loc[x]==df['DESC'].loc[y] and x!=y:
            remove_indexes.append(x)
            
            print('yes--  ',x,'-',df['DESC'].iloc[x],'  ----  ',y,'-',df['DESC'].iloc[y])
df=df.drop(remove_indexes)
df=(df.reset_index()).drop('index',axis=1)

#converting to tagged document for Doc2Vec

tagged_data=[TaggedDocument(words=word_tokenize(sentence.lower()),tags=[sentence.lower()]) for sentence in df['DESC']]
df['TAGGED_DATA']=tagged_data

#tagged data contains TaggedDocuments with word having the tokens and the tag as the same as the cleaned sentence,so that we can identidy it later

max_epochs = 100
vec_size = 100
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
                
model.build_vocab(tagged_data)
for epoch in range(max_epochs):
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha


#to get a sentence vector,use the cleaned form of the sentence as an index into model.docvecs
#to get vector for df['DESC'][0] use model.docvecs[df['DESC'][0]] or model[df['DESC'][0]]

#obtaining the vectors x and y
def create_XY(df_temp):
    df_temp=(df_temp.reset_index()).drop('index',axis=1)
    N=len(df_temp)
    X=[]
    Y=[]
    for i in range(N):
        X.append(model.docvecs[df_temp['DESC'][i]])
        cls=df_temp['CLASS'][i]
        if cls=='101':
            cls=[1,0,0,0,0]
        elif cls=='104':
            cls=[0,1,0,0,0]
        elif cls=='501':
            cls=[0,0,1,0,0]
        elif cls=='701':
             cls=[0,0,0,1,0]
        else:    
             cls=[0,0,0,0,1]
        Y.append(cls)        
    X=np.array(X)
    Y=np.array(Y)
    return X,Y

#training the NN
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.1)
X_train,Y_train=create_XY(train)
X_test,Y_test=create_XY(test)

MLP=MLPClassifier(hidden_layer_sizes=(70,35,),tol=0.001,max_iter=5000)
MLP=MLP.fit(X_train,Y_train)        
print('Accuracy for MLP  ')
print(MLP.score(X_test,Y_test))


sv3= DecisionTreeClassifier(max_depth=7)
sv3= sv3.fit(X_train,Y_train)
print('Accuracy for Decision Tree  ')
print(sv3.score(X_test,Y_test))

sv4= KNeighborsClassifier(5)
sv4= sv4.fit(X_train,Y_train)
print('Accuracy of KNN  ')
print(sv4.score(X_test,Y_test))

#print(X_train.shape)
#print(Y_train.shape)
Y_train = np.argmax(Y_train, axis=1)
sv2= SVC(gamma=2, C=4.5)
sv2= sv2.fit(X_train,Y_train)
print('Accuracy for RBF SVM   ')
#print(X_test.shape)
#print(Y_test.shape)
Y_test = np.argmax(Y_test, axis=1)
print(sv2.score(X_test,Y_test))


sv1= SVC(kernel="linear", C=0.025)
sv1= sv1.fit(X_train,Y_train)
print('Accuracy for linear SVM  ')
print(sv1.score(X_test,Y_test))

sv5= AdaBoostClassifier()
sv5= sv5.fit(X_train,Y_train)
print('Accuracy of AdaBoost  ')
print(sv5.score(X_test,Y_test))
