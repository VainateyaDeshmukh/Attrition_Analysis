import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import re
import nltk
import time
df = pd.read_csv("D:\\BA_College\\3rd Sem Assignment\\airline-sentiment.csv")
df.head()

len(df.index)
#Visualization
colors =sns.color_palette("hls",10)
pd.Series(df["airline_sentiment"].value_counts().plot(kind ="bar",color=colors, figsize=(8,6),
                                                      fontsize=10,rot=0, title ="Sentiment"))

airline = pd.crosstab(df.airline, df.airline_sentiment)
airline

percentage =airline.apply(lambda a:a /a.sum()*100, axis=1)
percentage

start_time=time.time()
#remove words which are starts with @ symbols
df['text'] = df['text'].map(lambda x:re.sub('@\w*','',str(x)))
#remove special characters except [a-zA-Z]
df['text'] = df['text'].map(lambda x:re.sub('[^a-zA-Z]',' ',str(x)))
#remove link starts with https
df['text'] = df['text'].map(lambda x:re.sub('http.*','',str(x)))
end_time = time.time()

#total time consume to filter data
end_time-start_time

df['text'].head()
#Converting to lower case
df['text'] = df['text'].map(lambda x:str(x).lower())
df['text'].head(2)


from nltk.corpus import stopwords
corpus =[]
#Remove stopwords from comments
remove=df['text'].map(lambda x:corpus.append(' '.join([word for word in str(x).strip().split()
                                                          if not word in set(stopwords.words('english'))])))
corpus[:6]

x =pd.DataFrame(data=corpus, columns=['comments'])
x.head()

y=df['airline_sentiment'].map({'negative':0,'positive':1})
y.head(4)

# Spliliting the data fro train test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
#token_patten
# 2 for word length greater than 2>=
vector = TfidfVectorizer(stop_words='english',sublinear_tf=True,strip_accents='unicode'
                         ,analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1,1),
                         max_features=30000)

X_train_word_feature = vector.fit_transform(X_train['comments']).toarray()
X_test_word_feature = vector.transform(X_test['comments']).toarray()
print(X_train_word_feature.shape,X_test_word_feature.shape)

#Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
classifier = LogisticRegression()
classifier.fit(X_train_word_feature,y_train)
y_pred = classifier.predict(X_test_word_feature)
cm = confusion_matrix(y_test,y_pred)
acc_score = accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred),'\n',cm,'\n',acc_score)

#To determine probability of negative or positive comment
y_pred_prob = classifier.predict_proba(X_train_word_feature)
y_pred_prob[:10]
