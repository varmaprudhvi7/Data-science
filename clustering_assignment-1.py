import pandas as pd
import matplotlib.pylab as plt 
import numpy as np
#for normalization 
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

df_norm = norm_func(crime.iloc[:,1:])
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 
type(df_norm)

#p = np.array(df_norm) # converting into numpy array format 
help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(df_norm) 
cluster_labels=pd.Series(h_complete.labels_)
crime['clust']=cluster_labels 
# creating a  new column and assigning it to new column 
crime_mean = crime.iloc[:,0:5].groupby(crime.clust).mean()
crime.head()
# where here crime rate is high cluster 0>cluster1>cluster2>cluster3