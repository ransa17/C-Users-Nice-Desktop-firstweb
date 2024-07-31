
"""

import pandas as pd

salary=pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Salary%20Data.csv')

salary.head()

salary.info()

salary.describe()

salary.tail()

salary.columns

salary.shape

y=salary['Salary']
x=salary[['Experience Years']]

x.shape

y.shape

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

x_train

from sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(x_train,y_train)

model.intercept_

model.coef_

y_pred=model.predict(x_test)

x_test

from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)

