# -*- coding: utf-8 -*-
"""AI_Mental_Fitness_Tracker.ipynb


## Importing Data
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the datasets
df1 = pd.read_csv('/content/mental-and-substance-use-as-share-of-disease (1).csv')
df2= pd.read_csv('/content/prevalence-by-mental-and-substance-use-disorder.csv')

"""### DATA Visualization"""

dataset=pd.merge(df1,df2)
dataset.head()

#set axis
dataset.set_axis(['Country','Code','Year','DALY','Schizophrenia','Bipolar disorder','Eating disorders',' Anxiety disorders','Drug use disorders','Depressive disorders','Alcohol use disorders'], axis='columns', inplace='True')

# Feature Engineering
# Select relevant features for mental fitness tracking
features = [ 'Schizophrenia','Bipolar disorder','Eating disorders',' Anxiety disorders','Drug use disorders','Depressive disorders','Alcohol use disorders']

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,6))
sns.heatmap(dataset.corr(),annot=True,cmap='coolwarm')
plt.plot()

sns.pairplot(dataset, corner= True)
plt.show()

mean=dataset['DALY'].mean()
mean

import plotly.express as px
fig=px.pie(dataset,values='DALY',names='Year')
fig.show()

#yearwise variation in Different countries mental fitness
fig=px.line_3d(dataset,x='Year',y='DALY',z='Country',color='Country' ,markers=True,template='plotly_dark')
fig.show()

"""# **Encoding the Categorical values**"""

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in dataset.columns:
    if dataset[i].dtype == 'object':
        dataset[i]=l.fit_transform(dataset[i])


dataset.head()

"""# **Assigning feautre of Matrix and Dependent Variable**"""

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

"""# **Implementing the Regression Algorithms and choose which one is better based on results**

# **1)Linear regression**
"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

"""# **Spliting the Dataset into Training set and Test set**"""

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

#predicting the value

y_pred=lr.predict(x_test)

#evaluating the model

from sklearn.metrics import r2_score,mean_squared_error
print("Results for Linear Regression:\n1)Mean Square Error={}\n2)R-Square Score={}".format(mean_squared_error(y_test,y_pred),r2_score(y_test,y_pred)))

"""# **2)SVM Regression**"""

from sklearn.svm import SVR
svr=SVR()
svr.fit(x_train,y_train)

#predicting the value

y_pred=svr.predict(x_test)

#evaluating the model

from sklearn.metrics import r2_score,mean_squared_error
print("Results for SVM Regression:\n1)Mean Square Error={}\n2)R-Square Score={}".format(mean_squared_error(y_test,y_pred),r2_score(y_test,y_pred)))

"""# **3)Decision Tree Regression**"""

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(x_train,y_train)

#predicting the value

y_pred=dtr.predict(x_test)

#evaluating the model

from sklearn.metrics import r2_score,mean_squared_error
print("Results for Decision Tree Regression:\n1)Mean Square Error={}\n2)R-Square Score={}".format(mean_squared_error(y_test,y_pred),r2_score(y_test,y_pred)))

"""# **4)Random Forest Regression**"""

from sklearn.ensemble import RandomForestRegressor
rfr= RandomForestRegressor(n_estimators=10,random_state=42)
rfr.fit(x_train,y_train)

#predicting the value

y_pred=rfr.predict(x_test)

#evaluating the model

from sklearn.metrics import r2_score,mean_squared_error
print("Results for Decision Tree Regression:\n1)Mean Square Error={}\n2)R-Square Score={}".format(mean_squared_error(y_test,y_pred),r2_score(y_test,y_pred)))

"""# **Conclusion:**

# **Random Forest Regression works** well on both train and test sets with r2 score of 0.99.

# As well as **Decision Tree Regression** also works well on both train and test set with r2 score of 0.98.

# **Predicting the value from the model using Random Forest Regression**
"""

np.random.seed(range(0,100))
print("Welcome to Mental Fitness Tracker!\nFill the detail to check your mental fitness!")
country=l.fit_transform([input('Enter Your country Name:')])
year=int(input("Enter the Year:"))
schi=(float(input("Enter your Schizophrenia rate in % (it not enter 0):")))*100
bipo_dis=(float(input("Enter your Bipolar disorder rate in % (it not enter 0):")))*100
eat_dis=(float(input("Enter your Eating disorder rate in % (it not enter 0):")))*100
anx=(float(input("Enter your Anxiety rate in % (it not enter 0):")))*10
drug_use=(float(input("Enter your Drug Usage rate in per year % (it not enter 0):")))*100
depr=(float(input("Enter your Depression rate in % (it not enter 0):")))*10
alch=(float(input("Enter your Alcohol Consuming rate per year in % (it not enter 0):")))*100

prediction=rfr.predict([[country,year,schi,bipo_dis,eat_dis,anx,drug_use,depr,alch]])
print("Your Mental Fitness is {}%".format(prediction))
print("Bye...!")
