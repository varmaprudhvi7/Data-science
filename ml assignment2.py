import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#-----------------Feature Engineering-------------------
startup=pd.get_dummies(startup, columns=["State"], prefix=["Statename"])
startup.rename(columns={'Statename_California':'california','Statename_New York':'newyork','Statename_Florida':'florida'},inplace=True)
startup.rename(columns={'R&D Spend':'rdspend','Marketing Spend':'marketingspend'},inplace=True)
startup.corr()
sns.pairplot(startup)
startup.columns
#-------------------MODEL BUILDING------------------
import statsmodels.formula.api as smf
model1=smf.ols('Profit ~ rdspend+Administration+marketingspend+california+florida+newyork',data=startup).fit()
model1.summary()
#here adjusted r^2 value =0.945 but pvalue is higher for marketing spend and administration
# so here we have check and either remove or make changes in it
import statsmodels.api as sm
sm.graphics.influence_plot(model1)
startup_new=startup.drop(startup.index[[48,49,46]],axis=0)
# by removies maximum influenced rows there no required change in the p value so go with vif to remove column
model2=smf.ols('Profit ~ rdspend+Administration+marketingspend+california+florida+newyork',data=startup).fit()
model2.summary()
pred = model2.predict(startup_new[['rdspend','Administration','marketingspend','california','florida','newyork']])
pred
# calculating VIF's values of independent variables
rsq_rd = smf.ols('rdspend~Administration+marketingspend+california+florida+newyork',data=startup_new).fit().rsquared  
vif_rd = 1/(1-rsq_rd)
#vif value for r&d spend= 2.7120
rsq_ad = smf.ols('Administration~rdspend+marketingspend+california+florida+newyork',data=startup_new).fit().rsquared  
vif_ad = 1/(1-rsq_ad)
#vif value for adminstration= 1.237739
rsq_ms = smf.ols('marketingspend~rdspend+Administration+california+florida+newyork',data=startup_new).fit().rsquared  
vif_ms = 1/(1-rsq_ms)
#vif value for marketing spend=2.709293
d1 = {'Variables':['rdspend','adminstration','marketspend'],'VIF':[vif_rd,vif_ad,vif_ms]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
sm.graphics.plot_partregress_grid(model2)
final_ml= smf.ols('Profit ~ rdspend+marketingspend+california+florida+newyork',data = startup_new).fit()
final_ml.params
final_ml.summary()
#here after removing adminstration we get adjusted r^2 =0.956
#and p value is significant
profit_pred = final_ml.predict(startup_new)
import pylab          
import scipy.stats as st
# Checking Residuals are normally distributed
st.probplot(final_ml.resid_pearson, dist="norm", plot=pylab)
plt.hist(final_ml.resid_pearson) # Checking the standardized residuals are normally distributed
plt.scatter(profit_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
# here the errors are randomly distrubuted

# so the best fit line is profit(y)=38694.507+r&d spend(0.74474)+marketing spend(0.03251)+california(13409.89556)+florida(12703.56)+newyork(12581.041)