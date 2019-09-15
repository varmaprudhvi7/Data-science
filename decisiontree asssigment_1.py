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
Fraud_new['is_train'] = np.random.uniform(0, 1, len(Fraud_new))<= 0.80
Fraud_new['is_train']
#spliting data inta train and test data
train,test = Fraud_new[Fraud_new['is_train'] == True],Fraud_new[Fraud_new['is_train']==False]
from sklearn.model_selection import train_test_split
train,test = train_test_split(Fraud_new,test_size = 0.3)

#------------------Model building------------------

from sklearn.tree import  DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
preds1=model.predict(train[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)
pd.crosstab(train[target],preds1)

#from sklearn.tree import DecisionTreeClassifier
#dtree=DecisionTreeClassifier()
#dtree.fit(target,predictors)

# Accuracy = train
np.mean(train.taxable == model.predict(train[predictors]))

# Accuracy = Test
np.mean(preds==test.taxable) 
#here we are getting the accuracy of the train in 1.0 and test is 0.644 it is very bad model we can us ensemble techniques like bagging and boosting 
#techniques to get best results
# and here the accuracys are continously changing


from sklearn.tree import export_graphviz
export_graphviz(model, out_file='tree.dot', 
                feature_names = predictors,
                class_names = target,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = ' ')

