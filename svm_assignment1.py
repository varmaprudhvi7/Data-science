import pandas as pd 
import numpy as np 
import seaborn as sns
forest=forest.drop("month",axis=1)
forest=forest.drop("day",axis=1)

forest.columns.get_loc("size_category")

cols = forest.columns.tolist()
cols = cols[28:] + cols[:28]
forest = forest[cols]  

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(forest,test_size = 0.3)

train_X = train.iloc[:,1:]
train_y = train.iloc[:,0]

test_X  = test.iloc[:,1:]
test_y  = test.iloc[:,0]

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) #0.9935 accuracy

model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 0.9487

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) #accuracy 0.74358




from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
plot_decision_regions(X=train_X, 
                      y=train_y,
                      clf=model_linear, 
                      legend=2)

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel(X.columns[0], size=14)
plt.ylabel(X.columns[1], size=14)
plt.title('SVM Decision Region Boundary', size=16)