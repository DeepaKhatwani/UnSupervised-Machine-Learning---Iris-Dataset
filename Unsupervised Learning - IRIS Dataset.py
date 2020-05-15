"""
Unsupervised learning is a machine learning technique, where you do not need 
to supervise the model. Instead, you need to allow the model to work on its 
own to discover information. It mainly deals with the unlabelled data. 

Here, are prime reasons for using Unsupervised Learning:

Unsupervised machine learning finds all kind of unknown patterns in data.


###############################################################################
Clustering: An unsupervised learning technique
Clustering is an important concept when it comes to unsupervised learning. 
It mainly deals with finding a structure or pattern in a collection of 
uncategorized data. Clustering algorithms will process your data and find 
natural clusters(groups) if they exist in the data. You can also modify how 
many clusters your algorithms should identify. It allows you to adjust the 
granularity of these groups. 

###############################################################################

Some applications of unsupervised machine learning techniques are:

- Clustering automatically split the dataset into groups base on their 
    similarities
- Anomaly detection can discover unusual data points in your dataset. It is 
    useful for finding fraudulent transactions
- Association mining identifies sets of items which often occur together in 
    your dataset
- Latent variable models are widely used for data preprocessing. Like reducing 
    the number of features in a dataset or decomposing the dataset into multiple 
    components
    
Some good links to see more resources:
    
- https://www.guru99.com/unsupervised-machine-learning.html
- https://towardsdatascience.com/unsupervised-learning-and-data-clustering-eeecb78b422a
- https://theappsolutions.com/blog/development/unsupervised-machine-learning/
- 
"""
#import the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

###############################################################################
#Simple example

from sklearn.cluster import KMeans 

import seaborn as sns

df1 = sns.load_dataset('iris')
print(type(df1))

#Data Exploration
df1.head()
df1.tail()
df1.shape
df1.info()
df1.groupby('species').describe().T
### Feature sleection for the model

#Considering only 2 features (Annual income and Spending Score) and no Label available
flower = {'setosa':0, 'versicolor':1,'virginica':2}

#df1['species'] = df1.species.map(flower)
df1.head()

X= df1.iloc[:, [0,1,2,3]].values
Y_Org = df1.iloc[:, [-1]].values
print(Y_Org)

from sklearn.cluster import KMeans
inertia=[]

#we always assume the max number of cluster would be 10

###get max no of clusters

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

    #inertia_ is the formula used to segregate the data points into clusters

#Visualizing the ELBOW method to get the optimal value of K 
plt.plot(range(1,11), inertia, '-bx')
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

#last elbow comes at k=3
"""
no matter what range we select ex- (1,21) also i will see the same behaviour 
but if we chose higher range it is little difficult to visualize the ELBOW
that is why we usually prefer range (1,11)
Finally we got that k=4
"""

#Model Build
kmeansmodel = KMeans(n_clusters= 3, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)
print(y_kmeans)
"""
This use case is very common and it is used in BFS industry(credit card) and 
retail for customer segmenattion.
"""

#Visualizing all the clusters 

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1') #versicolor
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') # setosa
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3') #virginica
plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of flowers')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()


y_pred_n=[] 
for i in range (len(y_kmeans)):
    if(y_kmeans[i]==0): #for ever 0 it will append Iris-setosa in the
        y_pred_n.append('versicolor')     
    elif (y_kmeans[i]==1): # for ever 1 it will append Iris-virginica in
        y_pred_n.append('setosa')
    else: # for ever 1 it will append Iris-versicolor in the y_pred_n 
         y_pred_n.append('virginica') 

print(y_pred_n)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(Y_Org, y_pred_n))
print(confusion_matrix(Y_Org, y_pred_n))
print(classification_report(Y_Org, y_pred_n))