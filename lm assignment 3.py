import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

#------------------FEATURE ENGINEERING-------------

plt.hist(delivery["Delivery Time"])
plt.hist(delivery["Sorting Time"])
delivery.rename(columns={'Delivery Time':'deltime','Sorting Time':'sortime'},inplace=True)
#scatter plot
plt.scatter(x=delivery['deltime'],y=delivery['sortime'],color=('red'))
#correlation
np.corrcoef(delivery['deltime'],delivery['sortime'])
#0.8259 not so strongly correlated

#----------------MODEL BUILDING----------------------

#-model1
#y=deltime)
model1=smf.ols('deltime~sortime',data=delivery).fit()
model1.summary()
#------------------0.682 r square value----------------
pred1=model1.predict(pd.DataFrame(delivery['sortime']))
res=delivery.deltime-pred1
sqres=res*res
mse=np.mean(sqres)
rmse=np.sqrt(mse)
rmse
model1.conf_int(0.05)
#rmse value=2.79165
#model2
model2=smf.ols('np.log(deltime)~sortime',data=delivery).fit()
model2.summary()
#.711 r square value
pred2=model2.predict(pd.DataFrame(delivery['sortime']))
res=delivery.deltime-pred2
sqres=res*res
mse=np.mean(sqres)
rmse=np.sqrt(mse)
rmse
#rmse value = 14.795516
#model3
model3=smf.ols('np.sqrt(deltime)~sortime',data=delivery).fit()
model3.summary()
#0.704 r square value
pred3=model3.predict(pd.DataFrame(delivery['sortime']))
res=delivery.deltime-pred3
sqres=res*res
mse=np.mean(sqres)
rmse=np.sqrt(mse)
rmse
#rmse value=13.523348
#getting residuals of entire data
student_resid = model2.resid_pearson 
student_resid
plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
# here errors are randomly distrubuted
#-----------predicted value vs actual values-------------
plt.scatter(x=pred2,y=delivery.deltime);plt.xlabel("Predicted");plt.ylabel("Actual")
plt.hist(model2.resid_pearson)

#------------model2 best fited line is given below------

#------delivery time(y)=(2.7727+0.2066(sorting time^2))---