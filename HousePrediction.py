#House prediction using housing data set containing housing prices in Ames iowa, there are extensive number of features
    #To begin I will:
        #Import the data with pandas, remove any null values, and one hot encode categoricals
        #Split the data into train and test sets
        #Log transform skewed features
        #Scaling can de attempted, although it can be interesting to see how well regularization works without scaling features

# %% 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pickle 
import pandas as pd
import matplotlib.pyplot as plt
import skillsnetwork
import sklearn


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, SGDRegressor
from sklearn.preprocessing import MinMaxScaler

# %%

#Import the data

print('\nPreparing to read local ames housing sales data set\n')

data_ames_housing = pd.read_csv('Ames_Housing_Sales.csv')

print("Data set download check:\n", data_ames_housing.head(10))

# %%

#create a list of categorical data and one-hot encode. Pandas one-hot encoder works well with data that is defined as categorical 

one_hot_encode_cols = data_ames_housing.dtypes[data_ames_housing.dtypes == object] #Filtering by string categoricals
one_hot_encode_cols = one_hot_encode_cols.index.tolist() #List of categorical fields

#Here we see another way of one hot encoding
#Encode there columns as categoricals so one hot encoding works on split data (if desired)
for col in one_hot_encode_cols:
    data_ames_housing[col] = pd.Categorical(data_ames_housing[col])

#Do the one hot encoding 
data_ames_housing = pd.get_dummies(data_ames_housing, columns = one_hot_encode_cols)

#Nest split the data in train and test split sets
train, test = train_test_split(data_ames_housing, test_size = 0.3, random_state = 42)

#There are a number of columns that have skewed features - a log transformation can be applied to them. That includes the predictor (SalePrice)
#However let's keep that one as is 

#Create a list of floeat columns to check for skewing 
mask = data_ames_housing.dtypes == float 
float_cols = data_ames_housing.columns[mask]

skew_limit = 0.75
skew_vals = train[float_cols].skew()

skew_cols = (skew_vals
             .sort_values(ascending = False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {0}'.format(skew_limit)))

print('\nSkew columns:\n', skew_cols)

# %%

#Transform all the columns where the skew is greater that 0.75, excluding SalePrice

print('\nPreparing for a graphic demostration\n')
field = 'BsmtFinSF1'
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10,5))
train[field].hist(ax = ax_before)
train[field].apply(np.log1p).hist(ax = ax_after)
ax_before.set(title = 'before np.log1p', ylabel = 'frequency', xlabel = 'value')
ax_after.set(title = 'after np.log1p', ylabel = 'frequency', xlabel = 'value')
fig.suptitle('Field  "{}" '.format(field))

# %%
#We can see it's a little bit better

for col in skew_cols.index.tolist():
    if col == 'SalePrice':
        continue
    train[col] = np.log1p(train[col])
    test[col] = test[col].apply(np.log1p) #Same thing

#Separate feature from predictor
feature_cols = [x for x in train.columns if x !='SalePrice']
X_train = train[feature_cols]
y_train = train['SalePrice']

X_test = test[feature_cols]
y_test = test['SalePrice']


#RMSE function that takes in trurh and prediction values and returns the root mean squared error
def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))

#Fit a basic linear regression model
#Print the root mean square error for this model
#Plot the predicted vs actual sale price based on the model 

linear_regression = LinearRegression().fit(X_train, y_train)

linear_regression_rmse = rmse(y_test, linear_regression.predict(X_test))

print('\nlinear regression rmse\n', linear_regression_rmse)

# %%

f = plt.figure(figsize=(6,6))
ax = plt.axes()

ax.plot(y_test, linear_regression.predict(X_test), marker = 'o', ls = '', ms = 3.0)

lim = (0, y_test.max())

ax.set(xlabel='Actual Price',
       ylabel = 'Predicted Price',
       xlim = lim,
       ylim = lim,
       title = 'Linear Regression Results')


# %%
#Ridge regression uses L2 normalization to reduce the magnitude of the coefficients. This can be helpful in situations where there is high variance.
#Fit a regular (non-cross-validated) Ridge model to a range of alpha values and plot the RMSE

ridge_alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]

ridge_cv = RidgeCV(alphas=ridge_alphas, cv=4).fit(X_train, y_train)

ridge_cv_rmse = rmse(y_test, ridge_cv.predict(X_test))

print('\nRidge alpha: ', ridge_cv.alpha_,'\nRdige rmse: ', ridge_cv_rmse)

# %%

#Much like ridge_cv function there is also lasso_cv function that uses L1 regularization function and cross-validation
#L1 will selectively shrink some coefficients

lasso_alphas =  np.array([1e-5, 5e-5, 0.0001, 0.0005])

lasso_cv = LassoCV(alphas=lasso_alphas, max_iter=100, cv=3).fit(X_train, y_train)

lasso_cv_rmse = rmse(y_test, lasso_cv.predict(X_test))

print('\nlasso_cv alpha: ', lasso_cv.alpha_, '\nlasso_cv rmse: ', lasso_cv_rmse)

#Can now determine how many of these features remain non-zero
print('\nOf {} coefficients, {} are non-zero with Lasso'.format(len(lasso_cv.coef_), len(lasso_cv.coef_.nonzero()[0])))

# %%

#Now I am gonna try with some elastic net, with the same alphas as in lasso, and l1_ratios between 0.1 and 0.9

l1_ratios = np.linspace(0.1, 0.9, 9)

elastic_net_cv = ElasticNetCV(alphas=lasso_alphas, l1_ratio=l1_ratios, max_iter=100).fit(X_train, y_train)
elastic_net_cv_rmse = rmse(y_test, elastic_net_cv.predict(X_test))

print('\nElastic Net CV alpha: ', elastic_net_cv.alpha_, '\nElastic Net CV l1 ratio: ', elastic_net_cv.l1_ratio_, '\nElastic Net CV rmse: ', elastic_net_cv_rmse)

#Comparing the RMSE calculation from all models is easiest in a table
rmse_vals = [linear_regression_rmse, ridge_cv_rmse, lasso_cv_rmse, elastic_net_cv_rmse]

labels = ['Linear', 'Ridge', 'Lasso', 'Elastic Net']

rmse_df = pd.Series(rmse_vals, index = labels).to_frame()
rmse_df.rename(columns = {0:'RMSE'}, inplace = 1)
print('\nRmse df: \n', rmse_df)

# %%

#We can also make a plot of actual vs predicted housing prices as before
f = plt.figure(figsize=(6,6))
ax = plt.axes()

labels = ['Ridge', 'Lasso', 'ElasticNet']

models = [ridge_cv, lasso_cv, elastic_net_cv]

for mod, lab in zip(models, labels):
    ax.plot(y_test, mod.predict(X_test), marker = 'o', ls = '', ms = 3.0, label = lab)

leg = plt.legend(frameon = True)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(1.0)

ax.set(xlabel = 'Actual Price',
       ylabel = 'Predicted Price',
       title = 'Linear Regression Results')

# %%

#I want to explore the Stochastic gradient descent
#SGD is very sensitive to scaling 
#Fit a stochastic gradient descent model without a regularization penalty
#Fit a stochastic gradient descent model with each three penalties (L1, L2, ElasticNet) using the parameter values determined by cv above
#Do not scale the data before fitting the model
#Compare the results to those obtained without using stochastic gradient descent

model_parameters_dict = {
    'Linear': {'penalty': None},
    'Lasso' : {'penalty': 'l2', 'alpha': lasso_cv.alpha_},
    'Ridge' : {'penalty': 'l1', 'alpha': ridge_cv_rmse},
    'ElasticNet': {'penalty': 'elasticnet', 'alpha': elastic_net_cv.alpha_, 'l1_ratio': elastic_net_cv.l1_ratio_}

}

new_rmses = {}
for modellabel, parameters in model_parameters_dict.items():
    #Following notation passes the dict items as args
    SGD = SGDRegressor(**parameters)
    SGD.fit(X_train, y_train)
    new_rmses[modellabel] = rmse(y_test, SGD.predict(X_test))

rmse_df['RMSE-SGD-learingrate'] = pd.Series(new_rmses)
print('\nNew rmse sgd\n', rmse_df)
# %%

#Notice how high the eror values are, this algorithm is diverging. This can be due to scaling and or learning rate being too high
#Adjust the learning rate and see what happens

model_parameters_dict = {
    'Linear': {'penalty': None},
    'Lasso': {'penalty': 'l2',
           'alpha': lasso_cv.alpha_},
    'Ridge': {'penalty': 'l1',
           'alpha': ridge_cv_rmse},
    'ElasticNet': {'penalty': 'elasticnet', 
                   'alpha': elastic_net_cv.alpha_,
                   'l1_ratio': elastic_net_cv.l1_ratio_}
                   }

new_rmses = {}
for modellabel, parameters in model_parameters_dict.items():
    #Followint the notation passes the dict items as arguments
    SGD = SGDRegressor(eta0=1e-7, **parameters)
    SGD.fit(X_train, y_train)
    new_rmses[modellabel] = rmse(y_test, SGD.predict(X_test))

rmse_df['RMSE-SGD-learningrate'] = pd.Series(new_rmses)
rmse_df
# %%

#Now scale the training data and try again

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


new_rmses = {}

for modellabel, parameters in model_parameters_dict.items():
    SGD = SGDRegressor(**parameters)
    SGD.fit(X_train_scaled, y_train)
    new_rmses[modellabel] = rmse(y_test, SGD.predict(X_test_scaled))

rmse_df['RMSE-SGD-scaled'] = pd.Series(new_rmses)
rmse_df
#%%