import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#-------------CLEANING DATA---------------
wine.describe()
wine_data=wine.iloc[:,1:]
# we are removing the one column which is not usefull
from sklearn.preprocessing import scale
wine_data=scale(wine_data)
# here data is normalized
from sklearn.decomposition import PCA
pca=PCA(n_components=13)
pca_values=pca.fit_transform(wine_data)
# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]
# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1
plt.plot(var1,color="red")
# plot between PCA1 and PCA2 
x =pca_values[:,0]
y =pca_values[:,1]
z =pca_values[:,2]
plt.scatter(x,y,color=["red","blue"])
#---------------here to check the correlation between them we used scatter plot---------------#
#------------------- Hierarchical Clustering-----------------
new_df = pd.DataFrame(pca_values[:,0:3])

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch
z = linkage(new_df, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0., 
    leaf_font_size=8.,
)
#------------its hard to interprut the dendrogram for large data sets------
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(new_df) 
#here we rae adding columns to data set
cluster_labels=pd.Series(h_complete.labels_)
wine['cluster']=cluster_labels 
#--------------------k means clustering---------------------

from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_df)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

model=KMeans(n_clusters=3) 
model.fit(new_df)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine['k means-clust']=md
#---------------now i checked with column 1 type we find more similar ---------------------
