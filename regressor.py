import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

"""
This file takes a csv file containing sample data of many functions.
Each function contains multiple x variables, and one y value.
80% of the sample data is used to train each machine learning mode (e.g. linear regression and random forest methods),
while 20% is used to test the model's prediction accuracy.
Then the accuracy of each model is put into a table and graphed.
"""

df = pd.read_csv("Machine_Learning_Regressors\delaney_solubility_with_descriptors.csv")

#seperate df into x and y variables
y=df['logS']

#axis=1 so df.drop works through columns; if axis=0 then works through rows
#save x as df except for the logS column data
x=df.drop('logS',axis=1)

#seperating data into training set (80% of data typically) and a testing set (20% of data typically),
#where the model is tested on the remaining 20% to see if it's accurate based off what it was trained on
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)


#build model

##Linear Regression Method 

#training the model
lr = LinearRegression()
lr.fit(X_train,y_train)

#using the model, try to predict the training set y values and testing set y values
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

#testing model performance
lr_train_mse = mean_squared_error(y_train,y_lr_train_pred)
lr_train_r2 = r2_score(y_train,y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test,y_lr_test_pred)
lr_test_r2 = r2_score(y_test,y_lr_test_pred)

lr_results = pd.DataFrame(['Linear Regression',lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2]).transpose()
lr_results.columns=('Method','Training MSE','Training R2','Test MSE','Test R2')



##Random Forest Method

#training the model

rf = RandomForestRegressor(max_depth=2,random_state=100)
rf.fit(X_train,y_train)
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

#testing model performance
rf_train_mse = mean_squared_error(y_train,y_rf_train_pred)
rf_train_r2 = r2_score(y_train,y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test,y_rf_test_pred)
rf_test_r2 = r2_score(y_test,y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest',rf_train_mse,rf_train_r2,rf_test_mse,rf_test_r2]).transpose()
rf_results.columns=('Method','Training MSE','Training R2','Test MSE','Test R2')


#model comparision
#axis=0 stacks the arrays row wise
df_models = pd.concat([lr_results,rf_results],axis=0).reset_index(drop=True)
#df_models.reset_index(drop=True)
print(df_models)


#data visualization of prediction results for linear regression
plt.figure(figsize=(5,5))
plt.scatter(x=y_train,y=y_lr_train_pred,c="#7CAE00",alpha=0.3)

z=np.polyfit(y_train,y_lr_train_pred,1)
p=np.poly1d(z)

plt.plot(y_train,p(y_train),'#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')
plt.show()


