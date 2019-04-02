# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:17:14 2019

@author: Dhawal
"""

import json
import re
import string
import math
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer


json_file=open('Dataset for Detection of Cyber-Trolls.json')

df=pd.DataFrame(columns=['content','label'])


c=0
for line in json_file:
    json_temp=json.loads(line)
    df.loc[c]=[json_temp['content'],int(json_temp['annotation']['label'][0])]
    c+=1
    
    
df= df.dropna() # remove missing values
df=df.drop_duplicates()
df=(df.reset_index()).drop('index',axis=1)
df = df[df.content.apply(lambda x: x !="")]
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

    #print(text)    
    text = text.split()
    token_count+=len(text)
    #stemmer = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    #stemmed_words = [stemmer.stem(word) for word in text]
    #text = " ".join(stemmed_words)
    lemmatized_words = [wordnet.lemmatize(word, pos='v') for word in text]
    text = " ".join(lemmatized_words)

    return text

df['content'] = df['content'].map(lambda x: clean_text(x))
avg_words_per_sentence=math.ceil(token_count/len(df))


df=(df.reset_index()).drop('index',axis=1)


df=(df.reset_index()).drop('index',axis=1)
n=len(df)
remove_indexes=[]
for x in range(n):
    for y in range(x+1,n):
        if df['content'].loc[x]==df['content'].loc[y] and x!=y:
            remove_indexes.append(y)
            
            print(x,'-',df['content'].loc[x],'  ----  ',y,'-',df['content'].loc[y])
        else:
            print('-',end='')
df=df.drop(remove_indexes)
df=(df.reset_index()).drop('index',axis=1)

tagged_data=[TaggedDocument(words=word_tokenize(sentence.lower()),tags=[sentence.lower()]) for sentence in df['content']]
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
    print('iteration {0}'.format(epoch))
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
        X.append(model.docvecs[df_temp['content'][i]])
        Y.append(df_temp['label'][i])        
    X=np.array(X)
    Y=np.array(Y)
    return X,Y


#training the NN
    
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.1)
X_train,Y_train=create_XY(train)
X_test,Y_test=create_XY(test)

MLP=MLPClassifier(hidden_layer_sizes=(50,),tol=0.001,max_iter=5000)
MLP=MLP.fit(X_train,Y_train)        

MLP.score(X_test,Y_test)

def predict(english_tweet):
    temp_x=model.infer_vector(doc_words=word_tokenize(clean_text(english_tweet)),steps=100,alpha=0.025)
    return MLP.predict(temp_x.reshape(1,-1))

err=0
a=0
for i in range(len(X_test)):
    pred_y=MLP.predict(X_test[i].reshape(1,-1))
    if pred_y==1:
        a+=1
    if pred_y!=Y_test[i]:
        err+=1
        
        