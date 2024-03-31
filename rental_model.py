
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#  Start coding 
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

# Further split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Create a Lasso model
lasso = Lasso(alpha=0.3, random_state=9)

# Train the model and access the coefficients
lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_

# Perform feature selection by choosing columns with positive coefficients
X_lasso_train = X_train.iloc[:, lasso_coef > 0]
X_lasso_test = X_test.iloc[:, lasso_coef > 0]

# Run OLS models on lasso chosen regression
ols = LinearRegression()
ols.fit(X_lasso_train, y_train)
pred = ols.predict(X_lasso_test)
mse_lin_reg_lasso = mean_squared_error(y_test, pred)

# Random Forest model

param_dist = {'n_estimators': np.arange(1,101,1),
              'max_depth': np.arange(1,11,1)}

# Creata a random forest regressor
rf = RandomForestRegressor()

# Use random search to find the best hyperparameters
rand_rearch = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist,
                                 cv = 5,
                                 random_state=9)

# Fit the random search to the train data
rand_rearch.fit(X_train, y_train)

# Best hyperparameters
hyper_params = rand_rearch.best_params_

# Run the random forest on the chosen hyperparameters
rf_hyper = RandomForestRegressor(n_estimators=hyper_params['n_estimators'],
                                 max_depth=hyper_params['max_depth'],
                                 random_state=9)

rf_hyper.fit(X_train, y_train)
rf_hyper_pred = rf_hyper.predict(X_test)
mse_rf = mean_squared_error(y_test, rf_hyper_pred)

print("The mean squared error for OLS models on lasso chosen regression model: {}".format(mse_lin_reg_lasso))
print("The mean squared error for the random forest model: {}".format(mse_rf))
