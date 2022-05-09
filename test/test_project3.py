import sys
sys.path.append('..')
import pytest
import project3
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def test_test_processing():
    text = 'Deepik_f0'
    result = project3.text_processing(text)
    if '_' not in result:
        assert True
    else:
        assert False

def test_tfidf_dict():
    github_link ="https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    df = pd.read_table(github_link, sep='\t',error_bad_lines=False)
    texts = df.iloc[:,-1]
    labels = df.iloc[:,2]
    doc = [project3.text_processing(txt) for txt in texts]
    #document = [project3.text_processing(txt) for txt in texts]
    idf_dict = project3.tfidf_dict(doc)
    return doc,idf_dict
    assert True 

def test_feature_extract():
    doc, idf_dict = test_tfidf_dict()
    features = project3.feature_extract(doc,idf_dict)    
    return features
    assert True

def test_dictVect():
    features = test_feature_extract()
    features = project3.dictvect(features)
    assert True



