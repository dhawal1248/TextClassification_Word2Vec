# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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




data_file="/home/samrat/anaconda3/yelp_labelled.csv"

df = pd.read_csv('/home/samrat/anaconda3/yelp_labelled.csv', sep = '\t', names = ['text','stars'], error_bad_lines = False)




df = df.dropna() # remove missing values
df = df[df.stars.apply(lambda x: x.isnumeric())]
df = df[df.stars.apply(lambda x: x !="")]
df = df[df.text.apply(lambda x: x !="")]
def clean_text(text):
    
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
    #stemmer = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    #stemmed_words = [stemmer.stem(word) for word in text]
    #text = " ".join(stemmed_words)
    
    lemmatized_words = [wordnet.lemmatize(word, pos='v') for word in text]
    text = " ".join(lemmatized_words)

    return text

df['text'] = df['text'].map(lambda x: clean_text(x))

tagged_data=[TaggedDocument(words=word_tokenize(sentence.lower()),tags=[sentence.lower()]) for sentence in df['text']]
#tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

#df['TAGGED_DATA']=tagged_data

max_epochs = 100
vec_size = 100
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.0025,
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

model.save("d2v.model")
print("Model Saved")

#print(model)

# Most similar from an inferred vector


#inferred_vector = model.infer_vector(doc_words=words, alpha=0.025, min_alpha=0.0001, steps=150)

#similar_doc = model.docvecs.most_similar(positive=[inferred_vector], topn=10)

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data


test_data = word_tokenize("Waiters are rude, In short will never return".lower())
v1 = model.infer_vector(test_data)
v1= v1.reshape(1, -1)
#print(v1.shape)
#print(v1)

#print("V1_infer", v1)

# to find most similar doc using tagss




#print(model.docvecs.most_similar(512,topn=10))
      


def create_XY(df_temp):
    df_temp=(df_temp.reset_index()).drop('index',axis=1)
    N=len(df_temp)
    X=[]
    Y=[]
    for i in range(N):
        X.append(model.docvecs[df_temp['text'][i]])
        #print(model.docvecs[df_temp['text'][i]])
        cls=df_temp['stars'][i]
        if cls=='0':
            cls=[1,0]
        else :
            cls=[0,1]
        
        Y.append(cls)        
    X=np.array(X)
    Y=np.array(Y)
    return X,Y

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



train, test = train_test_split(df, test_size=0.1)
X_train,Y_train=create_XY(train)
X_test,Y_test=create_XY(test)
#print(X_train.shape)
#print(Y_train.shape)


MLP=MLPClassifier(hidden_layer_sizes=(50),tol=0.001,max_iter=50000)
MLP=MLP.fit(X_train,Y_train)        
print('Accuracy of MLP   ')
print(MLP.score(X_test,Y_test))

v2=[0,0]
v2=MLP.predict(v1)
print (v2)


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
sv2= SVC(gamma=2, C=0.025)
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



#def read_input(input_file):
#    """This method reads the input file which is in gzip format"""
"""    
    logging.info("reading file {0}...this may take a while".format(input_file))
    
    with gzip.open ('/home/samrat/anaconda3/reviews_data.txt.gz', 'rb') as f:
        for i, line in enumerate (f): 

            if (i%10000==0):
                logging.info ("read {0} reviews".format (i))
            # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess (line)

# read the tokenized reviews into a list
# each review item becomes a serries of words
# so this becomes a list of lists
documents = list (read_input (data_file))
logging.info ("Done reading data file")    


model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)
    
w1 = "dirty"
model.wv.most_similar (positive=w1)"""