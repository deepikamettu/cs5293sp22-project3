import warnings
import numpy as np
import pandas as pd
from glob import glob
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
warnings.filterwarnings("ignore")

def text_processing(text):
    text = text.lower()
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'_', ' ', text)  
    text = re.sub(r"\'",' ',text)
    text = re.sub(r"\,",' ',text)
    text = re.sub(r"\.",' ',text)
    text = re.sub(r'\s+',' ',text)   
    text = re.sub(r'\b[a-z]{1,2}\b','',text)   
    text = re.sub(r'^[a-z]\s+',' ',text)       
    text = re.sub(r'\s+[a-z]\s+',' ',text)     
    text = re.sub(r'\s+',' ',text)          
    text = text.strip()
    return(text)

def tfidf_dict(document):
    doc = document
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit_transform(doc)
    idf = tfidf_vectorizer.idf_
    idf_dict = dict(zip(tfidf_vectorizer.get_feature_names(), idf))
    return idf_dict

def feature_extract(document,idf_dict):
    features=list()
    doc = document
    gitLink="https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    df = pd.read_table(gitLink, sep='\t',error_bad_lines=False)
    texts = df.iloc[:,-1]
    labels = df.iloc[:,2]
    for e,sent in enumerate(doc):
        feat_dict = {}
        name_len = len(labels[e])
        name_space = labels[e].count(' ')
        sent_len = len(sent)
        words = word_tokenize(sent)
        for j in range(len(words)):
            if u"\u2588" in words[j]:
                key = j
        if key-1 >= 0:
            left = words[key-1]
            if left in idf_dict.keys():
                left_idf = idf_dict[left]
            else:left_idf=0  
            
        else: left_idf = 0
        
        if not key+1 >= len(words):
            right = words[key+1]
            if right in idf_dict.keys():
                right_idf = idf_dict[right]
            else: right_idf=0
        else: right_idf = 0
        
        feat_dict['name_length'] = name_len
        feat_dict['name_spaces'] = name_space
        feat_dict['sent_length'] = sent_len
        feat_dict['left_idf'] = left_idf
        feat_dict['right_idf'] = right_idf
        
        features.append(feat_dict)
    return features

def dictvect(data):
    dvect=DictVectorizer()
    X=dvect.fit_transform(data).toarray()
    return X

if __name__ == '__main__':  
    gitLink="https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    df = pd.read_table(gitLink, sep='\t',on_bad_lines='skip')
    texts = df.iloc[:,-1]
    labels = df.iloc[:,2]
    doc = [text_processing(txt) for txt in texts]
    idf_dict = tfidf_dict(doc)
    features = feature_extract(doc,idf_dict)
    features = dictvect(features)
    le = LabelEncoder() # assigning int value to each name
    target = le.fit_transform(labels) 
    train_idx = df.index[df.iloc[:,1].isin(['training','validation'])].tolist()
    test_idx = df.index[df.iloc[:,1].isin(['testing','test'])].tolist()
    x_train, x_test = features[train_idx], features[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]
    #x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.11,shuffle=True)
    dt = DecisionTreeClassifier()
    dt.fit(x_train,y_train)
    dt.score(x_train,y_train)
    dt.score(x_test,y_test)
    prediction = dt.predict(x_test)
    print('        ----------Evaluation Report Of Classes-------------')
    #print('Accuarcy:',accuracy_score(y_test, prediction))
    print('Precision: {:,.2f}'.format(precision_score(y_test, prediction, average='weighted')))
    print('Recall-score: {:,.2f}'.format(recall_score(y_test, prediction, average='micro')))
    print('F1-score: {:,.2f}'.format(f1_score(y_test, prediction, average='macro')))
    print(le.inverse_transform(prediction))
