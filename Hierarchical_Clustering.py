import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Importing data for analysis
df = pd.read_csv("D:\\BA_College\\3rd Sem Assignment\\Attrition.csv")
df.head()
df.shape

X = df.iloc[:,[4,7]].values

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendogram")
plt.xlabel("Years at Company")
plt.ylabel("Monthly Income")

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='ward')
y_hc=hc.fit_predict(X)
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.title('(Hierarchical Clustering Model)')
plt.xlabel('Monthly Income')
plt.ylabel('Year at Company')
plt.show()