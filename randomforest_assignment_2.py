import pandas as pd
import matplotlib.pyplot as plt
#changing the variable names
Fraud.rename(columns={'Taxable.Income':'Taxable','Marital.Status':'martial','City.Population':'citypop','Work.Experience':'workexp'},inplace=True)
#here creating the dummy variables for taxable
df = pd.DataFrame(Fraud,columns=['Taxable'])
df.loc[df.Taxable<= 30000,'taxable'] = 'risky' 
df.loc[df.Taxable > 30000,'taxable'] = 'good' 
df_new=df.drop("Taxable",axis=1)
Fraud_new=Fraud.drop("Taxable",axis=1)
Fraud_new=Fraud_new.join(df_new)

#creating dummy variables for rest other variables
Fraud_new=pd.get_dummies(Fraud_new,columns=["Undergrad","Urban"],prefix=["undergo","urban"])
Fraud_new=Fraud_new.drop("undergo_NO",axis=1)
Fraud_new=Fraud_new.drop("urban_NO",axis=1)
#creating dummy variables for maratial status

df1 = pd.DataFrame(Fraud,columns=['martial'])
df1.loc[df1.martial=="Single",'Martial'] = 1 
df1.loc[df1.martial=="Married",'Martial'] = 2
df1.loc[df1.martial=="Divorced",'Martial'] = 3 
Fraud_new=Fraud_new.join(df1.Martial)
Fraud_new=Fraud_new.drop("martial",axis=1)
#Fraud_new=Fraud_new.drop("is_train",axis=1)

#to know the count
Fraud_new.taxable.value_counts()

cols = Fraud_new.columns.tolist()
cols = cols[-4:] + cols[:-4]
import numpy as np
#colnames = list(cols.columns)
predictors =cols[1:]
target = cols[0]

X = Fraud_new[predictors]
Y = Fraud_new[target]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")

rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_outputs_ # Number of outputs when fit performed
rf.oob_score_ 
rf.predict(X)

Fraud_new['rf_pred'] = rf.predict(X)
cols = ['rf_pred','taxable']
Fraud_new[cols].head()
Fraud_new["taxable"]


from sklearn.metrics import confusion_matrix
confusion_matrix(Fraud_new['taxable'], Fraud_new['rf_pred']) # Confusion matrix
pd.crosstab(Fraud_new['taxable'],Fraud_new['rf_pred'])
print ("Accuracy",(476+116)/(476+8+116))
# accuraccy=0.9866666