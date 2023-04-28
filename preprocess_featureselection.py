import pandas as pd
from collections import Counter
import json
import tqdm
import logging
from nltk import untag
import pymystem3
from pymystem3 import Mystem
import nltk
import string
import json, urllib
import pandas as pd
import numpy as np
import matplotlib

# Read dataset
data = pd.read_csv('lenta_data.csv')
data.head()
data.shape 

# tidy text
pattern = '|'.join(['[A-Za-z]', '[^\w\s]', '\d+', '\n', '\s+'])
data['tidy_text'] = data['text'].str.replace(pattern, ' ')
data['tidy_text'] = data['tidy_text'].str.lower()


# lemmatization
docs = list(data['tidy_text'])
len(docs)
m = Mystem()

lem_text = []
for i in tqdm.tqdm(docs):
    try:
        lemmas = m.lemmatize(i)
        lem = ''.join(lemmas)
        lem_text.append(lem)
    except:
        Exception

data['lem_text'] = lem_text

pattern = '|'.join(['\n', '\s+'])
data['lem_text'] = data['lem_text'].str.replace(pattern, ' ')

# stopwords
from nltk.corpus import stopwords
stop = stopwords.words('russian')

data['lem_text'] = data['lem_text']\
    .apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

data['lem_text'][1]

# days of the week and months
daysmonths = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье', 'январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь', 'год']

data['lem_text'] = data['lem_text']\
    .apply(lambda x: ' '.join([word for word in x.split() if word not in (daysmonths)]))

data['lem_text'][1]


import nltk.tag

# Tagging parts of speech
# Rule: subject + subject, adj + subject, subject alone

grammar = ('''
    AS: {<A=m><S>}
    SS: {<S><S>}
    S1: {<S>}
    ''')

chunkParser = nltk.RegexpParser(grammar)

lem_text = list(data['lem_text'])

dfs = []
i = 0
for text in tqdm.tqdm(lem_text):
    i = i + 1
    num_text = "text {}".format(i)
    tagged = nltk.pos_tag(nltk.word_tokenize(text), lang='rus')
    tree = chunkParser.parse(tagged)
    as_tag = []
    ss_tag = []
    s_tag = []
    for subtree in tree.subtrees():
        if subtree.label() == 'AS': 
            as_tag.append(" ".join(untag(subtree)).lower())
        if subtree.label() == 'SS': 
            ss_tag.append(" ".join(untag(subtree)).lower())
        if subtree.label() == 'S1': 
            s_tag.append(" ".join(untag(subtree)).lower())
    as_tag_df = pd.DataFrame({'term': as_tag})
    as_tag_df['tag'] = 'as_tag'
    
    ss_tag_df = pd.DataFrame({'term': ss_tag})
    ss_tag_df['tag'] = 'ss_tag'
    
    s_tag_df = pd.DataFrame({'term': s_tag})
    s_tag_df['tag'] = 's_tag'

    df = as_tag_df.append(ss_tag_df)
    df = df.append(s_tag_df)
    df['num_text'] = num_text
    
    dfs.append(df)

final_df = pd.concat(dfs)

df['term'].nunique()

final_df.head(10)
final_df.shape #(440221, 3)
final_df.groupby(['num_text', 'term']).count().sort_values('tag',ascending=False)

data = data.reset_index().drop('index', axis=1)

data['num_text'] = data.index + 1
data['num_text'] = 'text ' + data['num_text'].astype(str)

data['id'] = data['num_text'].map(lambda x: x.lstrip('text '))
final_df['id'] = final_df['num_text'].map(lambda x: x.lstrip('text '))


# DATA FOR CHI-SQUARED FEATURE SELECTION
date_term = data[['id', 'year', 'topic']].merge(final_df, on='id')


data_final = date_term.groupby(['id', 'topic'], as_index=False).agg({'term': ' '.join})

#data_final.to_csv("~/Desktop/ic2s2_internet_regulation-master 2/text_for_svm.csv", index=False)



date_term = data[['id', 'year']].merge(final_df, on='id')


term_count = date_term.groupby(['year', 'term']).size().reset_index(name="Time")

date_term = date_term.merge(term_count, on=['year','term'])

date_term = date_term[['term', 'year', 'id', 'Time']]
date_term.columns = ['term', 'cat', 'id', 'count']

date_term.head()

date_term.sort_values('count')

date_term = date_term[date_term['count'] > 5] 


for y in date_term['cat'].unique():
    date_term[date_term['cat'] == y]\
    [['term', 'cat', 'id', 'count']]\
    .drop_duplicates()\
    .to_csv("~/Desktop/ic2s2_internet_regulation-master 2/terms/terms_{}.csv".format(int(y)), index=False)





# Feature selection
import os
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph.
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import tqdm

# loading data
df = pd.read_csv("~/Desktop/ic2s2_internet_regulation-master 2/text_for_svm.csv")
print(df.shape)
df.head()

# Create a new dataframe with two columns
df1 = df[['topic', 'term']].copy()
# Remove missing values (NaN)
df1 = df1[pd.notnull(df1['term'])]
# Renaming second column for a simpler name
print(df1.shape)
df1.head(3)

pd.DataFrame(df1.topic.unique()).values

# Create a new column 'category_id' with encoded categories 
df1['category_id'] = df1['topic'].factorize()[0]
category_id_df = df1[['topic', 'category_id']].drop_duplicates()

# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'topic']].values)
# New dataframe
df1.head()


# Creating term-document matrix with tf-idf values

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2,
                        ngram_range=(1, 2))

features = tfidf.fit_transform(df1.term).toarray()
labels = df1.category_id


# We use the chi-squared test to find the terms that are the most correlated with each of the categories:

N = 20
for topic, category_id in tqdm.tqdm(sorted(category_to_id.items())):
  features_chi2 = chi2(features, labels == category_id) # input: term-document matrix with tf-idf values and each topic
  indices = np.argsort(features_chi2[0]) # чтобы получить от наиболее коррелирующего к наименее
  feature_names = np.array(tfidf.get_feature_names_out())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("%s:" %(topic))
  print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
  print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))

