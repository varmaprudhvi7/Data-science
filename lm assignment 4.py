import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
empdata.rename(columns={'Salary_hike':'salhike','Churn_out_rate':'churn'},inplace=True)
plt.hist(empdata["salhike"])
plt.scatter(x=empdata['salhike'],y=empdata['churn'],color=('red'))
# by using z distrubution we made that data unitfree
from scipy import stats
zscore=stats.zscore(empdata)
zscore
columns_new=['salhike','churn']
empdata=pd.DataFrame(zscore, columns=columns_new)
np.corrcoef(empdata['salhike'],empdata['churn'])
#-------------------------data is now unit less-------------------
#correlation = -0.91172
import statsmodels.formula.api as smf
#------------model1-------------------
model1=smf.ols('churn~salhike',data=empdata).fit()
model1.summary()
#-----------r squared value= 0.831---------------
pred1=model1.predict(pd.DataFrame(empdata['salhike']))
res=empdata.churn-pred1
sqres=res*res
mse=np.mean(sqres)
rmse=np.sqrt(mse)
rmse
#---------------rmse value =0.41080-------------
#--------------model2--------------------
model2=smf.ols('churn~np.log(salhike)',data=empdata).fit()
model2.summary()
#-----------r squared value= 0.891---------------
pred2=model2.predict(pd.DataFrame(empdata['salhike']))
res=empdata.churn-pred2
sqres=res*res
mse=np.mean(sqres)
rmse=np.sqrt(mse)
rmse
#----------------rmse value=0.125099----------------
#-------------model3-----------------------
model3=smf.ols('np.log(churn)~salhike',data=empdata).fit()
model3.summary()
#-----------r squared value= 0.981---------------
pred3=model3.predict(pd.DataFrame(empdata['salhike']))
res=empdata.churn-pred3
sqres=res*res
mse=np.mean(sqres)
rmse=np.sqrt(mse)
rmse
#----------------rmse value=4.09419-----------------
#----------------model4-----------------------
model4=smf.ols('churn~np.sqrt(salhike)',data=empdata).fit()
model4.summary()
#-----------r squared value= 0.989---------------
pred4=model4.predict(pd.DataFrame(empdata['salhike']))
res=empdata.churn-pred4
sqres=res*res
mse=np.mean(sqres)
rmse=np.sqrt(mse)
rmse
#-------------rmse value=0.0390-----------
student_resid = model4.resid_pearson 
student_resid
plt.plot(model4.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
# here errors are randomly distrubuted
#-----------predicted value vs actual values-------------
plt.scatter(x=pred4,y=empdata.churn);plt.xlabel("Predicted");plt.ylabel("Actual")
plt.hist(model4.resid_pearson)

#------------best fited line is given below------
#---------churn out rate(y)=-0.1973+-0.8021(sqrt(salhike))-----------------

