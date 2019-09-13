import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#-------------FEATURE ENGINEERING---------------------
toyota.columns
toyota_new=toyota[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
toyota_new.columns
# --------here we are standralizing the values-----------
#zscore=stats.zscore(toyota_new)
#zscore
#toyota_new=["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]
#toyota_z=pd.DataFrame(zscore, columns=toyota_new)
toyota_new.corr()
sns.pairplot(toyota_new)
from scipy import stats
#--------building models--------------
import statsmodels.formula.api as smf
model1=smf.ols('Price ~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_new).fit()
model1.summary()
# where here adjusted r^2 value is 0.863 but doors and cc are insignificant
# so either we have to make changes or remove the coloumn
model2=smf.ols('Price ~ Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=toyota_new).fit()
model2.summary()
#--------------r^2 value 0.863-------------
import statsmodels.api as sm
sm.graphics.influence_plot(model2)
toyota_new=toyota_new.drop(toyota_new.index[[80]],axis=0)
model3=smf.ols('Price ~ Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=toyota_new).fit()
model3.summary()
#here by removing the 80th row now p value is significant
# here adjusted r^2 value is 0.869
#so now try to increase r^2 value using transformation
model4=smf.ols('np.sqrt(Price) ~ Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=toyota_z).fit()
model4.summary()
# here r^2 value is 0.663
# among them for model3 adj r^2 is 0.869 we consider it 
price_pred = model3.predict(toyota_new)
import pylab          
import scipy.stats as st
# Checking Residuals are normally distributed
st.probplot(model3.resid_pearson, dist="norm", plot=pylab)
# they are normally distrubuted
 # Checking the standardized residuals are normally distributed
plt.hist(model3.resid_pearson)
#scatter plot
plt.scatter(price_pred,model3.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
# here the errors are randomly distrubuted
# here the best fit line is price(y)=-6233.88+(-120.55)(Age_08_04)+(-0.0179)(KM)+39.278(HP)+-2.5108