import pandas as pd
import numpy as np
company.columns
colnames = list(company.columns)
predictors = colnames[1:8]
target = colnames[0]

company=pd.get_dummies(company,columns=["US","Urban"],prefix=["US","urban"])
company=company.drop("US_No",axis=1)
company=company.drop("urban_No",axis=1)

#here we create dummy variables for ShelveLoc
df2 = pd.DataFrame(company,columns=['ShelveLoc'])
df2.loc[df2.ShelveLoc=="Good",'shell'] = 1 
df2.loc[df2.ShelveLoc=="Bad",'shell'] = 2
df2.loc[df2.ShelveLoc=="Medium",'shell'] = 3 
company=company.join(df2.shell)
company=company.drop("ShelveLoc",axis=1)

# here we are creating dummy varables for Sales
df1 = pd.DataFrame(company,columns=['Sales'])
df1.loc[df1.Sales<4,'Sales'] = 1
df1.loc[(df1.Sales>=4) & (df1.Sales<8),'Sales'] = 2
df1.loc[(df1.Sales>=8) & (df1.Sales<=12),'Sales'] = 3
df1.loc[(df1.Sales>12) & (df1.Sales<17),'Sales'] = 4
#here we are joing the Sales dummy variables
company_new=company.drop("Sales",axis=1)
company_new=company_new.join(df1.Sales)


cols = company_new.columns.tolist()
predictors =cols[:10]
target = cols[10]

X = company_new[predictors]
Y = company_new[target]
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")

rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_outputs_ # Number of outputs when fit performed
rf.oob_score_ 
rf.predict(X)

company_new['rf_pred'] = rf.predict(X)
cols = ['rf_pred','Sales']
company_new[cols].head()
company_new["Sales"]


from sklearn.metrics import confusion_matrix
confusion_matrix(company_new['Sales'],company_new['rf_pred']) # Confusion matrix
pd.crosstab(company_new['Sales'],company_new['rf_pred'])

