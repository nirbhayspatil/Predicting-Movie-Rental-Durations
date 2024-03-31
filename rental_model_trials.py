import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Import any additional modules and start coding below
rental = pd.read_csv('C:/Users/patilni/OneDrive - Kantar/Desktop/Projects/Predicting Movie Rental Durations/rental_info.csv')
rental['rental_length'] = pd.to_datetime(rental['return_date']) - pd.to_datetime(rental['rental_date'])
rental['rental_length_days'] = rental['rental_length'].dt.days
rental.shape

## Adding dummay variables
rental['deleted_scenes'] = np.where(rental['special_features'].str.contains('Deleted Scenes'),1,0)
rental['behind_the_scenes'] = np.where(rental['special_features'].str.contains('Behind the Scenes'),1,0)

# Choose columns to drop
cols_to_drop = ['special_features', 'rental_length', 'rental_length_days', 'rental_date', 'return_date']

# Split into feature and target sets
X = rental.drop(cols_to_drop, axis=1)
y = rental['rental_length_days']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_pred)

svr = SVR()
svr.fit(X_test, y_test)
svr_pred = svr.predict(X_test)
svr_mse = mean_squared_error(y_test, svr_pred)

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)
rfr_mse = mean_squared_error(y_test, rfr_pred)

print("The mean squared error of LinearRegression is {a}, DecisionTreeRegressor is {b}, SVR is {c} and RandomForestRegressor is {d}".format(a=lr_mse, b=dt_mse, c=svr_mse, d=rfr_mse))

rfr.estimator_params

