import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame 
df_norm = norm_func(zoo.iloc[:,1:])
k = list(range(2,18))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
# here total number of clusters is 3
model=KMeans(n_clusters=3) 
model.fit(df_norm)
model.labels_ 
md=pd.Series(model.labels_) 
zoo['animal types']=md 
df_norm.head()
# now from here we are starting the knn 
zoo.rename(columns={'animal name':'ani'},inplace=True)
zoo=zoo.drop('ani',axis=1)

from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo,test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier as KNC
knn = KNC(n_neighbors= 3)

knn.fit(train.iloc[:,0:17],train.iloc[:,17])

# train accuracy 
train_acc = np.mean(knn.predict(train.iloc[:,0:17])==train.iloc[:,17]) # 94 %


# test accuracy
test_acc = np.mean(knn.predict(test.iloc[:,0:17])==test.iloc[:,17]) # 100%

acc = []
for i in range(3,50,2):
    knn = KNC(n_neighbors=i)
    knn.fit(train.iloc[:,0:17],train.iloc[:,17])
    train_acc = np.mean(knn.predict(train.iloc[:,0:17])==train.iloc[:,17])
    test_acc = np.mean(knn.predict(test.iloc[:,0:17])==test.iloc[:,17])
    acc.append([train_acc,test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")

plt.legend(["train","test"])
