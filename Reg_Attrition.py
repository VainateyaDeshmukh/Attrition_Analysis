import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Importing data for analysis
df = pd.read_csv("D:\\BA_College\\3rd Sem Assignment\\Attrition.csv")
df.head()
df.shape

X = df.iloc[:,[4]].values
Y = df.iloc[:,[3]].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.25,random_state=10)

#creating the model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_predict = reg.predict(x_test)

print('Accuracy on training data: {:.3f}'.format(reg.score(x_train,y_train)))
print('Accuracy on testing date: {:.3f}'.format(reg.score(x_test,y_test)))

print(reg.coef_)
print(reg.intercept_)

#scatter plot
plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.title('Reg_Plot')
plt.xlabel('Income')
plt.ylabel('Age')