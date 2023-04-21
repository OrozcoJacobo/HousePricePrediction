#House prediction using housing data set containing housing prices in Ames iowa, there are extensive number of features
    #To begin I will:
        #Import the data with pandas, remove any null values, and one hot encode categoricals
        #Split the data into train and test sets
        #Log transform skewed features
        #Scaling can de attempted, although it can be interesting to see how well regularization works without scaling features

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

#Import the data

print('\nPreparing to read local ames housing sales data set\n')

data_ames_housing = pd.read_csv('Ames_Housing_Sales.csv')

print("Data set download check:\n", data_ames_housing.head(10))

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

#Transform all the columns where the skew is greater that 0.75, excluding SalePrice

field = 'BsmtFinSF1'
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10,5))
train[field].hist(ax = ax_before)
train[field].apply(np.log1p).hist(ax = ax_after)
ax_before.set(title = 'before np.log1p', ylabel = 'frequency', xlabel = 'value')
ax_after.set(title = 'after np.log1p', ylabel = 'frequency', xlabel = 'value')
fig.suptitle('Field  "{}" '.format(field))