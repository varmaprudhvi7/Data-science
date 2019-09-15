import pandas as pd 
import numpy as np 
import seaborn as sns

sal_dummies = pd.get_dummies(salary[["workclass","occupation","education","maritalstatus","relationship","race","sex","native"]])
# Dropping the columns for which we have created dummies
salary.drop(["workclass","education","maritalstatus","occupation","relationship","race","sex","native"],inplace=True,axis = 1)
salary = pd.concat([salary,sal_dummies],axis=1)

salary["cat"] = 0
salary.loc[salary.Salary==" <=50K","cat"] = 1
salary.Salary.value_counts()
salary.cat.value_counts()
salary.drop(["Salary"],axis=1,inplace=True)

sal_dummies1 = pd.get_dummies(salarytest[["workclass","occupation","education","maritalstatus","relationship","race","sex","native"]])
# Dropping the columns for which we have created dummies
salarytest.drop(["workclass","education","maritalstatus","occupation","relationship","race","sex","native"],inplace=True,axis = 1)
salarytest = pd.concat([salarytest,sal_dummies1],axis=1)

salarytest["cat"] = 0
salarytest.loc[salarytest.Salary==" <=50K","cat"] = 1
salarytest.Salary.value_counts()
salarytest.cat.value_counts()
salarytest.drop(["Salary"],axis=1,inplace=True)

#due to larger dataset we are using PCA -dimension reduction



train_X = salary.iloc[:,:102]
train_y = salary.iloc[:,102]
test_X  = salarytest.iloc[:,102]
test_y  = salarytest.iloc[:,102]

from sklearn.svm import SVC
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 85.233