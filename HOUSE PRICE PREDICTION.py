#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[59]:


df_train = pd.read_csv(r"C:\Users\SRK\OneDrive\Desktop\house_prediction\train.csv")
df_test =  pd.read_csv(r"C:\Users\SRK\OneDrive\Desktop\house_prediction\test.csv")


# In[60]:


df_train.shape


# In[61]:


df_train.head(50)


# In[62]:


df_train.info()


# In[63]:


df_train.describe()


# ### In this Exploratory Data Analysis
#  <li>Finding th Missing Values
#  <li>Removing Nan
#  <li>Finding the Categorical Values

# In[64]:


## Here we will check the percentage of nan values present in each feature
## 1 -step make the list of features which has missing values
features_with_na=[features for features in df_train.columns if df_train[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(df_train[feature].isnull().mean(), 4),  ' % missing values')


# ### Observation:
# Checking whether the missing values have any relationship with SalePrice.If the feature has a Nan Value it is converted to 1 and otherwise it is 0

# In[65]:



for feature in features_with_na:
    data = df_train.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    # let's calculate the mean SalePrice where the information is missing or present
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()


# The df_train.copy() is used in the above cell and will be used in the upcoming cells.This means that any changes that are done in the data while manipulation may not get affected in the actual data.
# ### Observation:
# We can clearly see that there is relationship between Nan Values and Saleprice.Look at the diagram of GarageCond                 where the absesence of info on the featur lead to the decrease in the price of the house

# In[66]:


print("Id of Houses {}".format(len(df_train.Id)))


# ### Dropping the features that are helpful
# ### Finding the numerical Variables
# 
# We are dividing the numerical data into: 
# <li>Data with the info about YEARS.
# <li>Discrete variables.
# <li>Continuous variables.

# In[67]:


numerical_features = [feature for feature in df_train.columns if df_train[feature].dtypes != 'O']
print("No. of Numerical variables: ",len(numerical_features))


# ###  Observation:
# ### Temporal Variables:
# Of the above numerical values, there are 4 variables which have relation between them. They are called Temporal Variables

# In[68]:


year_col = [year for year in df_train if "Y" in year]
year_col


# In[69]:


for feature in year_col:
    print(feature,df_train[feature].unique())


# In[70]:


## Lets analyze the Temporal Datetime Variables
## We will check whether there is a relation between year the house is sold and the sales price

df_train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")


# ### Observation:
# In the above given graph, the prices are very low in the recent years than compared to the early years.This trend was started exactly in 2007. We need to do some more evalution to validate the above plot

# In[71]:


## Here we will compare the difference between All years feature with SalePrice

for feature in year_col:
    if feature!='YrSold':
        data=df_train.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# ### Discrete Variables:
# Discrete variables are countable in a finite amount of time. For example, you can count the change in your pocket. You can count the money in your bank account. You could also count the amount of money in everyone’s bank accounts.It might take you a long time to count that last item, but the point is—it’s still countable. 

# In[72]:


## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(df_train[feature].unique())<25 and feature not in year_col+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
print(discrete_feature)


# In[73]:


df_train[discrete_feature]


# ### Observation:
# EDA is all about finding the relationship between the dependent variable and Independent variables.So no matter what, we just go deeper and deeper.
# Now finding the realtionship between discrete variables and dependent variables.

# In[74]:


## Lets Find the realtionship between them and Sale PRice

for feature in discrete_feature:
    data=df_train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# ### Monotonic Relation: 
# The relation which is exponentially increasing between OverallQual and SalePrice is known as Monotonic Relation.

# ### Continuous Variables:
# Continuous variable alludes to the a variable which assumes infinite number of different values.The values are obtained by measuring.A continuous variable always takes the value in the given range like max. and min.
# 
# Now lets find the Continuous Varaibles in the given dataset
# 
#     

# In[75]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature + year_col + ['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))


# ### Observation:
# We need to find the relation between dependent Variable and these Continuous Features. Since the data is continuous we need to find the distribution between them. The Histograms are efficient in getting more info in this kind of scenarios.

# In[76]:


## Lets analyse the continuous values by creating histograms to understand the distribution

for feature in continuous_feature:
    data=df_train.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# In[77]:


#skewness and kurtosis
for feature in continuous_feature:
    print("Skewness: %f"% df_train[feature].skew())
    print("Kurtosis: %f"% df_train[feature].kurt())


# ### Observation:
# All the histograms are left-skewed.None of the graph is in Gausian Distribution AKA Normal Distribution. We need to convert all these non-gausian disribution graphs to a gausian distribution.The skewness of normal disribution must be "zero".

# ### How can we make a non-gausian distribution to Gausian Distribution:
# There are three ways to convert a non-gausian distribution to Gausian Distribution:
# <li>Squaring the feature.
# <li>Cubing the feature.
# <li>Logarithmic Transformation.
# 
# The logarithmic transformation is said to be more efficient than above it. So we go with that.
#     We are performing scatter plot between SalePrice and continuous feature. For that we need to perform the logarithmic transformation for every continuous feature along with SalePrice so that we can map each transformed feature with new transformed SalePrice itself.  

# In[78]:


## Performing logarithmic transformation


for feature in continuous_feature:
    data=df_train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()
        data[feature].hist(bins=25)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title(feature)
        plt.show()


# ### Observation: 
# Every plot in the result exhibits a monotonic relationship and also the gausian disribution with the independent feature excluding the plot of SalePrice with SalePrice.

# ### Chasing the Outliers:
# What defines an outlier?
# 
# An outlier is an observation that lies an abnormal distance from other values in a random sample from a population.Boxplots are good to explore about Outliers as they show you where the outlier is lying, like bewlow 25th percentile or aboove 100th percentile and etc.
# 
# #### Note:
# If your data is purely categorical, then you can not graph it on a boxplot as it measures quantitative data. If you want to graph purely categorical data, a bar graph is probably a better option because it compares amounts of different categories.Since the data is continuous, I am using Boxplots.

# In[79]:


for feature in continuous_feature:
    data=df_train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
        


# ### Observation:
# There are lot of outliers in each of the boxplots above.

# In[80]:


### Finding the categorical Features:
categorical_features=[feature for feature in df_train.columns if data[feature].dtypes ==  'O']
categorical_features


# In[81]:


##Finding the categories in each categorical features
for feature in categorical_features:
    print(' {}  {}'.format(feature,len(df_train[feature].unique())))


# ### Observation:
# For categorical features with 5 unique categories we use One-hot encoding.
# 
# Finding the relationship between dependent variable and Categorical features.

# In[82]:


for feature in categorical_features:
    data=df_train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# ### Observation: 
# There are different kinds of realtionship between each plot

# ### Data Leakage: 
# 
# Data leakage (or leakage) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction. This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.
# 
# To avoid data leakage we split the data into train data and test data
# 

# In[83]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df_train,df_train['SalePrice'],test_size=0.1,random_state=0)


# In[84]:


X_train.shape, X_test.shape


# ### Handling the Missing Values

# Going for Missing Values in Categorical Values

# In[85]:


## Let us capture all the nan values
## First lets handle Categorical features which are missing
features_nan=[feature for feature in df_train.columns if df_train[feature].isnull().sum()>1 and df_train[feature].dtypes=='O']

for feature in features_nan:
    print("{}: {}% missing values".format(feature,np.round(df_train[feature].isnull().mean(),4)))


# In[86]:


## Replace missing value with a new label
def replace_cat_feature(df_train,features_nan):
    data=df_train.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

dataset=replace_cat_feature(df_train,features_nan)

dataset[features_nan].isnull().sum()


# ### Observation:
# Replacing every Nan Value with the new label 'Missing' so that it becomes a new category itself.

# In[87]:


dataset.head()


# ### Going for Missing Values in Numerical Values

# In[88]:


## Now lets check for numerical variables the contains missing values
numerical_with_nan=[feature for feature in df_train.columns if df_train[feature].isnull().sum()>1 and df_train[feature].dtypes!='O']

## We will print the numerical nan variables and percentage of missing values

for feature in numerical_with_nan:
    print("{}: {}% missing value".format(feature,np.around(df_train[feature].isnull().mean(),4)))


# ### Observation: 
# There are only three variables with missing Values. Since the percentage of missing Values is very low, we can replace them with the Median which will be an apt replacement

# In[89]:


## Replacing the numerical Missing Values

for feature in numerical_with_nan:
    ## We will replace by using median since there are outliers
    median_value=dataset[feature].median()
    
    ## create a new feature to capture nan values
    dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
    dataset[feature].fillna(median_value,inplace=True)
    
dataset[numerical_with_nan].isnull().sum()


# ###  Observation:
# In the above cell, we are creating a new feature column that fills with '1' if there is any Nan value and fills with '0' if there is no Nan value. After that we are filling the old feature column with the Median. In this way, we can still have track of the missing values even though they are filled with median.

# In[90]:


dataset.head(50)


# ### Handling the Temporal Variables:
# 

# In[91]:


## Temporal Variables (Date Time Variables)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    dataset[feature]=dataset['YrSold']-dataset[feature]


# In[92]:


dataset[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()


# ### Observation: 
# Since the temporal variables are with 2013, 1965, 1874 and etc which is difficult to compute, we are converting all this into a better values by which we are subtracting each of them with feature 'Year Sold'

# ### Checking with Numerical Variables:
# 

# In[93]:


numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
for feature in numerical_features:
    print('{}  {}'.format(feature,dataset[feature].skew()))


# ### Observation:
# Some of the numerical features are skewed. So we need to make them follow Gausian Distribution. I have already mentioned the methods to remove skewness. 

# In[94]:


import numpy as np
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    dataset[feature]=np.log(dataset[feature])


# In[ ]:





# In[95]:


dataset.head()


# In[96]:


for feature in numerical_features:
    print('{}  {}'.format(feature,dataset[feature].skew()))


# ### Observation:
# Some of the numerical features are skewed. So we need to make them follow Gausian Distribution. I have already mentioned the methods to remove skewness. 

# In[97]:



num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    dataset[feature]=np.log(dataset[feature])


# In[98]:


dataset.head()


# ### Feature Scaling
# Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

# In[117]:


scaling_feature=[feature for feature in dataset.columns if feature not in ['Id','SalePerice'] ]
len(scaling_feature)


# In[118]:


scaling_feature


# In[121]:


feature_scale=[feature for feature in dataset.columns if feature not in ['Id','SalePrice']]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(dataset[feature_scale])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <h3>Let's Find out if their is any relationship between LotArea an SalePrice

# In[100]:


df_train['LotArea'].describe()


# In[101]:


df_train['SalePrice'].describe()


# In[102]:


# here we set the figure size to 15x8
plt.figure(figsize=(15, 8))
# plot two values price per lot size
plt.scatter(df_train.SalePrice, df_train.LotArea)
plt.xlabel("price ", fontsize=14)
plt.ylabel("lot size", fontsize=14)
plt.title("Scatter plot of price and lot size",fontsize=18)
plt.show()


# In[ ]:





# In[ ]:





# In[103]:


#histogram
sns.distplot(df_train['SalePrice']);


# In[104]:


#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# In[105]:


#scatter plot grlivarea/saleprice
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));


# In[106]:


#scatter plot totalbsmtsf/saleprice
data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));


# <h1>Relationship with Categorical Values</h1>
# 

# In[107]:


#box plot overallqual/saleprice
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[108]:


var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# <h1>Correlation Matrix and Heat Map

# In[109]:


corrmat = df_train.corr()
corrmat


# In[110]:


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[111]:


df_train.corr()


# In[112]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[113]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# In[114]:


#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...


# <H1>Standardizing the Data

# In[ ]:


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[ ]:


#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[ ]:


#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])


# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# In[ ]:


#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[ ]:


#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])


# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# In[ ]:


#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);


# In[ ]:


#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);


# In[ ]:


#convertnig categorical variables into dummy
df_train = pd.get_dummies(df_train)


# In[ ]:


df_train


# <h1>Fitting the Model

# In[ ]:


df_fit = df_train[['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','WoodDeckSF','OpenPorchSF','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']]
X = df_fit
y = df_train['SalePrice']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# If your RMSE is much higher than your MAE, it (loosely) means your error variance is high. It could mean that you probably have outliers in your data.If your RMSE is much higher than your MAE, it (loosely) means your error variance is high. It could mean that you probably have outliers in your data.

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)


# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)


# In[ ]:


print(lin_reg.intercept_)


# In[ ]:


y_pred = lin_reg.predict(X_test)


# In[ ]:





# <h1>Metrics to Validate Regression Results
# <h6>Mean Absolute Error: 
# 
#     
#     MAE is the average value of error in a set of predicted values, without considering direction. It ranges from 0 to inf., and lower value means better model. It is the simplest to understand regression error metric.
#     
# Root Mean Squared Error (RMSE): 
#     
#     RMSE is the square root of average value of squared error in a set of predicted values, without considering direction. It ranges from 0 to inf., lower means better model and it is always greater in magnitude than MAE. 

# In[ ]:


import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
import math

 
MSE = mean_squared_error(y_test, y_pred)
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:",RMSE)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # or mse**(0.5)  
r2 = metrics.r2_score(y_test,y_pred)

print("Results of sklearn.metrics:")
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)


# <h3>MAE is lower limit of RMSE. RMSE should always be higher than or equal to MAE.If your RMSE is much higher than your MAE, it (loosely) means your error variance is high. It could mean that you probably have outliers in your data.

# In[ ]:


pd.get_option("display.max_columns")


# In[ ]:


pd.set_option("display.max_columns", None)


# In[ ]:


print("Accuracy:", r2*100,'%')


# In[ ]:


df_fit.describe()


# In[ ]:


for i in df_train:
    if "Heating" in i:
        heating_col = i
        print(df_train[heating_col].describe)


# In[ ]:



    


# In[ ]:





# In[ ]:




