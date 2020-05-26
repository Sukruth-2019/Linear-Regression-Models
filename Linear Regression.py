#!/usr/bin/env python
# coding: utf-8

# ## Kaggle Competition for House Prices: Advanced Regression Techniques 

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
pd.pandas.set_option('display.max_columns',None)
pd.pandas.set_option('display.max_rows',None)


# In[2]:


get_ipython().system('pip install -U scikit-learn')


# In[3]:


get_ipython().system('pip install sklearn')


# In[2]:


from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression_model
from sklearn.preprocessing import MinMaxScaler


# ## For Train Data

# In[3]:


df=pd.read_csv('C:/Users/sukru/Downloads/train.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


features_with_na = [i for i in df.columns if df[i].isnull().sum()>1]
features_with_na
for i in features_with_na:
    print(i,np.round(df[i].isnull().mean(),4),'% of missing values')


# ### Since they are many missing values, we need to find the relationship between missing values and Sales Price
# 
# Ploting some diagram for this relationship
# 

# In[10]:


for feature in features_with_na:
    data = df.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature] = np.where(df[feature].isnull(), 1, 0)
    
    #Calculate the mean SalePrice where the information is missing or present
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()


# Here With  the relation between the missing values and the dependent variable is clearly visible.So We need to replace these nan values with something meaningful which we will do in the Feature Engineering section

# In[11]:


print('Id of houses {}'.format(df['Id'].count()))


# In[12]:


print('Id of houses {}'.format(len(df['Id'])))


# ### Numerical Variables

# In[13]:


numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
print('Total numerical features are {}'.format(len(numerical_features)))
df[numerical_features].head()


# #### Temporal Variables(Eg: Datetime Variables)
# 
# From the Dataset we have 4 year variables. We have extract information from the datetime variables like no of years or no of days. One example in this specific scenario can be difference in years between the year the house was built and the year the house was sold. 

# In[14]:


year_features=[i for i in numerical_features if 'Yr' in i or 'Year' in i]
print('Total Temporary Features are {}'.format(len(year_features)))
year_features


# ### Let's explore the content of these year variables

# In[15]:


for i in year_features:
    print(i,df[i].unique())


# ### We will check whether there is a relation between year the house is sold and the sales price

# In[16]:


## Lets analyze the Temporal Datetime Variables
df.groupby('YrSold')['SalePrice'].median().plot(marker='o')
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")
plt.show()


# In[17]:


## Here we will compare the difference between All years feature with SalePrice

for feature in year_features:
    if feature!='YrSold':
        data=df.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# In[18]:


## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables


# ####  Discrete Variable

# In[19]:


discrete_feature= [i for i in numerical_features if len(df[i].unique())<25 and i not in year_features+['Id']]
print('Total Discrete Features are {}'.format(len(discrete_feature))) 


# In[20]:


discrete_feature


# In[21]:


df[discrete_feature].head()


# #### Continuous Variable

# In[22]:


continuous_feature=[i for i in numerical_features if i not in discrete_feature+year_features+['Id']]
#continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print('Total Continuous Feature are {}'.format(len(continuous_feature)))


# In[23]:


continuous_feature


# In[24]:


df[continuous_feature].head()


# In[25]:


## Lets analyse the continuous values by creating histograms to understand the distribution

for feature in continuous_feature:
    data=df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# ### Outliers

# In[26]:


for feature in numerical_features:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
        
    


# ### Categorical Variables

# In[33]:


catergorical_features= [i for i in df.columns if df[i].dtypes == 'O']
#categorical_features=[feature for feature in df.columns if df[feature].dtypes=='O']

print('Total Catergorical Features are {}'.format(len(catergorical_features)))


# In[34]:


catergorical_features


# In[35]:


## To find out unique  in each Catergorical feature 
for feature in catergorical_features:
    print('The feature is {} and there are unique catergories are {}.'.format(feature,df[feature].nunique()))


# In[37]:


for feature in catergorical_features:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[224]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[147]:


df.shape


# ## Missing Value Treatment

# In[ ]:


df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[ ]:


df.drop(['Alley'],axis=1,inplace=True)


# In[ ]:


df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])


# In[ ]:


df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])


# In[ ]:


df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[ ]:


df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])


# In[ ]:


df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.drop(['Id'],axis=1,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[ ]:


df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')


# In[ ]:


df.dropna(inplace=True)


# In[38]:


df.shape


# In[38]:


df.head()


# ### Another Method of treating Missing Values

# In[39]:


catergorial_nan=[i for i in df.columns if df[i].isnull().sum()>1 and df[i].dtypes=='O']
for i in catergorial_nan:
    print('{} : {}% of missing values'.format(i,np.round(df[i].isnull().mean(),4)))


# In[40]:


## Replace missing value with a new label
def replace_cat(feature,catergorical_nan):
    data=df.copy()
    data[catergorical_nan]=data[catergorical_nan].fillna('Missing')
    return data


# In[41]:


df=replace_cat(df,catergorial_nan)
df[catergorial_nan].isnull().sum()


# In[42]:


df.head(50)


# In[43]:


df.shape


# In[44]:


numerical_nan=[i for i in df.columns if df[i].isnull().sum()>1 and df[i].dtypes!= 'O']
for i in numerical_nan:
    print('the feature is {} and {}% are the missing values.'.format(i,np.round(df[i].isnull().mean(),4)))


# In[45]:


250/1460, 81/1460


# In[46]:


## Treating missing value for the numerical features 
def replace_num(feature,numerical_nan):
    data=df.copy()
    data[numerical_nan]=data[numerical_nan].fillna(data[numerical_nan].median())
    return data


# In[47]:


df=replace_num(df,numerical_nan)
df[numerical_nan].isnull().sum()


# In[48]:


df.head(10)


# In[50]:


## Treating Datatime variable into Interger
yr_features=['YearBuilt','YearRemodAdd','GarageYrBlt']
for i in yr_features:
    df[i]=df['YrSold']-df[i]


# In[51]:


df[yr_features].head()


# ###  Numerical Variables
# From the EDA we could find the numerical variables are skewed we will perform log normal distribution.

# In[52]:


df.head()


# In[53]:


num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    df[feature]=np.log(df[feature])


# In[54]:


df.head()


# ### Handling Rare Categorical Feature
# 
# We will remove categorical variables that are present less than 1% of the observations

# In[55]:


cat_features=[i for i in df.columns if df[i].dtypes=='O']
cat_features


# In[56]:


for feature in cat_features:
    temp=df.groupby(feature)['SalePrice'].count()/len(df)
    temp_df=temp[temp>0.01].index
    df[feature]=np.where(df[feature].isin(temp_df),df[feature],'Rare_var')


# In[57]:


df.head(50)


# In[59]:


for feature in catergorical_features:
    labels_ordered=df.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df[feature]=df[feature].map(labels_ordered)


# In[60]:


df.head(10)


# In[61]:


scaling_feature=[feature for feature in df.columns if feature not in ['Id','SalePerice'] ]
len(scaling_feature)


# ###  Feature Scaling

# In[62]:


feature_scale=[i for i in df.columns if i not in ['Id','SalePrice']]

scaler=MinMaxScaler()
scaler.fit(df[feature_scale])


# In[63]:


scaler.transform(df[feature_scale])


# In[64]:


# transform the train and test set, and add on the Id and SalePrice variables
data=pd.concat([df[['Id','SalePrice']].reset_index(drop=True),pd.DataFrame(scaler.transform(df[feature_scale]),
                                                                           columns=feature_scale)],axis=1)


# In[66]:


data.head()


# In[67]:


data.shape


# In[265]:


data.to_csv('C:/Users/sukru/Downloads/X_train.csv',index=False)


# In[519]:


df=pd.read_csv('C:/Users/sukru/Downloads/X_train.csv')


# In[520]:


df.head()


# In[504]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[506]:


## Capture the dependent feature
y_train=df[['SalePrice']]


# In[507]:


## drop dependent feature from dataset
X_train=df.drop(['Id','SalePrice'],axis=1)


# In[509]:


### Apply Feature Selection
# first, I specify the Lasso Regression model, and I
# select a suitable alpha (equivalent of penalty).
# The bigger the alpha the less features that will be selected.

# Then I use the selectFromModel object from sklearn, which
# will select the features which coefficients are non-zero

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_train, y_train)


# In[510]:


feature_sel_model.get_support()


# In[513]:


# let's print the number of total and selected features

# this is how we can make a list of the selected features
selected_feat = X_train.columns[(feature_sel_model.get_support())]

# let's print some stats
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
#print('features with coefficients shrank to zero: {}'.format(
 #   np.sum(sel_estimator_coef_ == 0)))


# In[514]:


selected_feat


# In[600]:


df1=X_train[selected_feat]


# In[601]:


df1.head()


# In[524]:


df=pd.concat([df[['Id','SalePrice']],df1],axis=1)


# In[525]:


df.head()


# In[526]:


df.shape


# ### Separate train and test set

# In[528]:


# Let's separate into train and test set

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Id','SalePrice'],axis=1), df[['SalePrice']], test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape,y_train.shape,y_test.shape


# In[529]:


X_train.head()


# In[530]:


y_train.head()


# ### Machine Learning algorithm building

# #### AdaBoostRegressor

# In[555]:


from sklearn.ensemble import AdaBoostRegressor 
xgb_model = AdaBoostRegressor()
eval_set = [(X_test, y_test)]
xgb_model.fit(X_train, y_train)

pred = xgb_model.predict(X_train)
print('xgb train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = xgb_model.predict(X_test)
print('xgb test mse: {}'.format(mean_squared_error(y_test, pred)))


# ### Random Forest

# In[545]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

pred = rf_model.predict(X_train)
print('rf train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = rf_model.predict(X_test)
print('rf test mse: {}'.format(mean_squared_error(y_test, pred)))


# #### Support vector machine

# In[560]:


from sklearn.svm import SVR
SVR_model = SVR()
SVR_model.fit(X_train, y_train)

pred = SVR_model.predict(X_train)
print('SVR train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = SVR_model.predict(X_test)
print('SVR test mse: {}'.format(mean_squared_error(y_test, pred)))


# #### Regularised linear regression

# In[607]:


from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

pred = lin_model.predict(X_train)
print('linear train mse: {}'.format(mean_squared_error(y_train, pred)))
pred_test = lin_model.predict(X_test)
print('linear test mse: {}'.format(mean_squared_error(y_test, pred_test)))


# ### Submission to Kaggle

# In[ ]:


pred_ls = []
for model in [xgb_model, rf_model]:
    pred_ls.append(pd.Series(model.predict(X_test)))

pred = SVR_model.predict(X_test)
pred_ls.append(pd.Series(pred))

pred = lin_model.predict(X_test)
pred_ls.append(pd.Series(pred))

final_pred = pd.concat(pred_ls, axis=1).mean(axis=1)


# In[ ]:


temp = pd.concat([submission.Id, final_pred], axis=1)
temp.columns = ['Id', 'SalePrice']
temp.head()


# In[ ]:


temp.to_csv('submit_housesale.csv', index=False)


# In[ ]:




