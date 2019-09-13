import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#---------------FEATURE ENGINEERING-----------------
 #here we are creating dummy variablesfor cd                       
df_cd = pd.get_dummies(computer['cd'])
computer= pd.concat([computer, df_cd], axis=1)
computer.rename(columns={'yes':'cdyes'},inplace=True)
computer.drop(["cd","no"],axis=1,inplace=True)
#here we are creating dummy variables for premium
df_cd1 = pd.get_dummies(computer['premium'])
computer= pd.concat([computer, df_cd1], axis=1)
computer.rename(columns={"yes":'premiumyes'},inplace=True)
computer.drop(["premium1","no"],axis=1,inplace=True)
#due to insignificance we had removed multi column
computer.drop(["multi"],axis=1,inplace=True)
computer.corr()
sns.pairplot(computer)
#---------------building model------------------
import statsmodels.formula.api as smf
model1=smf.ols('np.log(price)~(speed+hd+ram+screen+cdyes+premiumyes+ads+trend)',data=computer).fit()
model1.params
model1.summary()
# here adjusted r^2 value is 0.780

model2=smf.ols('np.sqrt(price)~speed+hd+ram+screen+cdyes+premiumyes+ads',data=computer).fit()
model1.params
model2.summary()
#here adjudted r^2 is 0.534

model3=smf.ols('price~np.sqrt(speed+hd+ram+screen+cdyes+premiumyes+ads)',data=computer).fit()
model3.params
model3.summary()
#here adjusted r^2 value is 0.235

model4=smf.ols('(price)~np.log(speed+hd+ram+screen+cd+premium+ads)',data=computer).fit()
model4.params
model4.summary()
#here adjusted r^2 is 0.225

pred = model1.predict(computer)
pred
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model1)
# Observed values VS Fitted values
plt.scatter(computer.price,pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
# they represents model
# Residuals VS Fitted Values 
plt.scatter(pred,model1.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
# the residuals are distrubuted equally

#----------Normality plot for residuals
# histogram
plt.hist(model1.resid_pearson)
# the follow normal distrubution

#----------Best fitted line is given below------------
# price(y)=exponential(0.4080+0.0040(speed)+0.0003(hd)+0.0201(ram)+0.0559(screen)+2.2230(cd)+2.1850(premium)+0.0004(ads)+(-0.0218)trend)