import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Importing data for analysis
df = pd.read_csv("D:\\BA_College\\3rd Sem Assignment\\Attrition.csv")
df.head()
df.shape

#Data Pre-Processing
df.Attrition.value_counts()
df.Gender.value_counts()
df.columns
df.drop(columns=["Department","MonthlyIncome"], inplace=True)
df.isnull().sum()
#Data Exploration
sns.countplot(df['Gender'])
fig =plt.gcf()
fig.set_sizeinches(10,10)
plt.title('Gender')
sns.barplot(x="Gender",y="Age",data=df)
sns.barplot(x="Gender",y="Attrition",data=df)
sns.barplot(x="Gender",y="JobSatisfaction",data=df)
sns.barplot(x="Gender",y="TotalWorkingYears",data=df)
sns.barplot(x="Gender",y="YearsAtCompany",data=df)
#Data Pre-processing
df["Gender"].replace("Male",1,inplace=True)
df["Gender"].replace("Female",0,inplace=True)
#Model Building
x = df.drop(columns=['Attrition'])
y = df["Attrition"]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=10)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.fit_transform(X_test)
#Model KNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print('Accuracy on training data: {:.3f}'.format(knn.score(X_train,y_train)))
print('Accuracy on testing data: {:.3f}'.format(knn.score(X_test,y_test)))

#Confussion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

#Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(knn,x,y,cv=10)
print('{:.3f}'.format(accuracies.mean()))

#Pipeline

from sklearn.pipeline import make_pipeline
clf = make_pipeline(sc,knn)
accuracies = cross_val_score(clf,x,y,cv=10)
print('{:.3f}'.format(accuracies.mean()))
