import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
min(company.Sales)
max(company.Sales)
#here we created dummy variables for US and Urban
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
company_new['is_train'] = np.random.uniform(0, 1, len(company_new))<= 0.75
company_new['is_train']


train,test = company_new[company_new['is_train'] == True],company_new[company_new['is_train']==False]
from sklearn.model_selection import train_test_split
train,test = train_test_split(company_new,test_size = 0.2)



from sklearn.tree import  DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)


# Accuracy = train
np.mean(train.Sales == model.predict(train[predictors]))

# Accuracy = Test
np.mean(preds==test.Sales) 
