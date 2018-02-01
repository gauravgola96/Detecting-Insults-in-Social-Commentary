import pandas as pd

train = pd.read_csv("C:\\Users\\Gaurav_Gola\\Desktop\\Praxis\\text\\my work\\Project\\train1.csv")

test = pd.read_csv("C:\\Users\\Gaurav_Gola\\Desktop\\Praxis\\text\\my work\\Project\\test.csv")

train.head() # lables are insult 
# 1 = insult
# 0 = not insult


import seaborn as sn
import matplotlib.pylab as plt 
%matplotlib inline

train.Insult = train.Insult.astype("category")

print(train[train.Insult==0].count())
print(train[train.Insult==1].count())
# 0 - not insult - are more - 2898
# 1 - insult - are less - 1049 

# cleanning part will done on comments of both train and test datasets 
# separate comments of both train and test data sets 
train_label = train["Insult"]
train_comment = train["Comment"]

test_comment = test["Comment"]
train_label.shape



type(test_comment)

#merge train_comment and test_comment for cleaning purpose 

Data_to_clean = pd.concat([train_comment,test_comment],axis=0)
#axis = 0 -- row wise 
# axis = 1 - column wise - this will create problem while cleaning of data 

Data_to_clean.head()



type(Data_to_clean)

#Text Cleaning part .... 
# Tokenize 

import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
import re 

# error is coming in tokenizing
#Presence of special character such as // should be removed 
# otherwise nltk cannot identify where is the sentence is stopped or started

# Removing special character before tokenization 


CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

import contractions
from contractions import contractions_dict
from contractions import contractions_re_keys
import re

def remove_characters_before_tokenization(sentence,keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]' # add other characters here tocremove them
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    else:
        PATTERN = r'[^a-zA-Z0-9 ]' # only extract alpha-numeric characters
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence

Data_to_clean1 = [remove_characters_before_tokenization(i) for i in Data_to_clean]

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def expand_contractions(sentence, contraction_mapping):
    import re
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    
    def expand_match(contraction):
        
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                               if contraction_mapping.get(match)\
                               else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence

stopword_list = nltk.corpus.stopwords.words('english')


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        text = expand_contractions(text, CONTRACTION_MAP)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
    return normalized_corpus

normalized_data = normalize_corpus(corpus=Data_to_clean1,tokenize=False)

normalized_data

from nltk.corpus import wordnet


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens

Data_to_clean2 = remove_repeated_characters(normalized_data)
Data_to_clean2

Data=[]

for text in Data_to_clean2:
    Data.append(text)

Data

#splitting data back 

train_corpus = Data[:3947]
test_corpus = Data[3947:]



#feature Extraction
# TF-IDF

import sklearn
from sklearn.feature_extraction.text import CountVectorizer

def feat_extract(data,ngram_range):
    vectorizer = CountVectorizer(min_df=1,ngram_range=ngram_range)
    feature = vectorizer.fit_transform(data)
    return(vectorizer,feature)

train_vec,train_feat = feat_extract(data=train_corpus,ngram_range=(1,3))

train_vec.get_feature_names()

print(train_vec)
print(train_feat)

train_features = train_feat.todense()

test_vec,test_feat = feat_extract(data=test_corpus,ngram_range=(1,3))

test_vec.get_feature_names()

test_features = test_feat.todense()
test_features

from sklearn.feature_extraction.text import TfidfTransformer

def tfidf_transformer(matrix):
    transform = TfidfTransformer(norm='l2',smooth_idf=True,use_idf=True)
    tfidf_matrix = transform.fit_transform(matrix)
    
    return(transform, tfidf_matrix)

train_transform , train_matrix = tfidf_transformer(train_features)

train_final_feature = train_matrix.todense()

test_transform,test_matrix = tfidf_transformer(test_features)

test_final_feature = test_matrix.todense()

test_final_feature

test_final_feature.shape

import scipy
from scipy import sparse


test_final_feature.shape

# converting to sparse matrix
X_training,X_testing=sparse.csr_matrix(train_final_feature),sparse.csr_matrix(test_final_feature)

# inspecting the transformed data
type(X_training),type(X_testing), X_training.shape, X_testing.shape

type(train_label)

X_train = X_training[0:3157,0:14180]
X_test = X_training[790:,0:14180]
y_train = train_label[:3157]
y_test = train_label[790:]

X_train.shape

#train_label

#Modelling....
#Naive Bayes
from sklearn import cross_validation as cv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.cross_validation import cross_val_score

from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()

import numpy as np
data = np.array(train_final_feature)

#(X_train,X_test,Y_train,_Y_test) = train_test_split(X=X_training,y=np.array(train_label),test_size=0.33,random_state=42)

NB.fit(X=X_train,y=y_train)

cross_val_score(estimator=NB,X=X_test,y=y_test,cv=5)

NB_pred = NB.predict(X_test)

print(accuracy_score(y_true=y_test,y_pred=NB_pred))
      

print(accuracy_score(y_true=y_test,y_pred=NB_pred),"Accuracy")

print(f1_score(y_true=y_test,y_pred=NB_pred,average='weighted'),"F1_score")

print(recall_score(y_true=y_test,y_pred=NB_pred,average='weighted'),"recall_score/sensitivity")

print(precision_score(y_true=y_test,y_pred=NB_pred,average='weighted'),"precision_score")




#SVM


from sklearn.linear_model import SGDClassifier
SDG = SGDClassifier()

SDG.fit(X=X_train,y=y_train)

cross_val_score(estimator=SDG,X=X_test,y=y_test,cv=5)

SDG_pred = SDG.predict(X_test)

print(accuracy_score(y_true=y_test,y_pred=SDG_pred),"Accuracy")

print(f1_score(y_true=y_test,y_pred=SDG_pred,average='weighted'),"F1_score")

print(recall_score(y_true=y_test,y_pred=SDG_pred,average='weighted'),"recall_score/sensitivity")

print(precision_score(y_true=y_test,y_pred=SDG_pred,average='weighted'),"precision_score")
