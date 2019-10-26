
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm,skew,kurtosis
from scipy.special import boxcox1p


# In[2]:

train = pd.read_csv("train__updated.csv")
test = pd.read_csv("test__updated.csv")


# In[3]:

train.head()


# In[4]:

test.head()


# In[5]:

train_index = train['index']
test_index = test['index']


# In[6]:

train.drop("index", axis = 1, inplace = True)
test.drop("index", axis = 1, inplace = True)


# In[7]:

sns.distplot(train['wave_height'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['wave_height'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Wave height distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['wave_height'], plot=plt)
plt.show()


# In[8]:

#Log-transformation of the target variable. (We applied log transformation, 
# box-cox transformation and cube root transformation on the target variable, of which, log 
# transformation gave the best results).
train["wave_height"] = np.log1p(train["wave_height"]+300)

#Check the new distribution 
sns.distplot(train['wave_height'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['wave_height'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Wave height distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['wave_height'], plot=plt)
plt.show()


# In[9]:

#Data Correlation
corrmat = train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[10]:

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.wave_height.values
train.drop(['wave_height'], axis=1, inplace=True)
all_data = pd.concat((train, test)).reset_index(drop=True)
print("all_data size is : {}".format(all_data.shape))


# In[11]:

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# In[12]:

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()


# In[13]:

all_data.drop("release_no_primary", axis = 1, inplace = True)
all_data.drop("release_no_secondary", axis = 1, inplace = True)
all_data.drop("release_no_tertiary", axis = 1, inplace = True)
all_data.drop("release_status_indicator", axis = 1, inplace = True)
all_data.drop("intermediate_reject_flag", axis = 1, inplace = True)
all_data.drop("source_exclusion_flags", axis = 1, inplace = True)
all_data.drop("dup_check", axis = 1, inplace = True)
all_data.drop("dup_status", axis = 1, inplace = True)
all_data.drop("swell_direction", axis = 1, inplace = True)


# In[14]:

all_data.drop("swell_period", axis = 1, inplace = True)
all_data.drop("sst_measurement_method", axis = 1, inplace = True)
all_data.drop("year", axis = 1, inplace = True)
#all_data.drop("day", axis = 1, inplace = True)
all_data.drop("imma_version", axis = 1, inplace = True)
all_data.drop("attm_count", axis = 1, inplace = True)
all_data.drop("latlong_indicator", axis = 1, inplace = True)
all_data.drop("national_source_indicator", axis = 1, inplace = True)
all_data.drop("wind_direction_indicator", axis = 1, inplace = True)
all_data.drop("dpt_indicator", axis = 1, inplace = True)


# In[15]:

#all_data.drop("swell_direction", axis = 1, inplace = True)
#all_data.drop("wetbulb_temerature", axis = 1, inplace = True)
all_data.drop("dewpoint_temperature", axis = 1, inplace = True)
#all_data.drop("wetbulb_temerature", axis = 1, inplace = True)
all_data.drop("wbt_indicator", axis = 1, inplace = True)
all_data.drop("visibility", axis = 1, inplace = True)
all_data.drop("indicator_for_temp", axis = 1, inplace = True)
#all_data.drop("time indicator", axis = 1, inplace = True)


# In[16]:

all_data.drop("wind_direction_true", axis = 1, inplace = True)
all_data.drop("wind_speed_indicator", axis = 1, inplace = True)
all_data.drop("platform_type", axis = 1, inplace = True)
all_data.drop("nightday_flag", axis = 1, inplace = True)
all_data.drop("deck", axis = 1, inplace = True)
all_data.drop("source_id", axis = 1, inplace = True)
all_data.drop("wetbulb_temperature", axis = 1, inplace = True)
all_data.drop("time_indicator", axis = 1, inplace = True)
all_data.drop("id_indicator", axis = 1, inplace = True)
all_data.drop("present_weather", axis = 1, inplace = True)
all_data.drop("past_weather", axis = 1, inplace = True)
all_data.drop("lower_cloud_amount", axis = 1, inplace = True)
all_data.drop("total_cloud_amount", axis = 1, inplace = True)
all_data.drop("ship_speed", axis = 1, inplace = True)
all_data.drop("ship_course", axis = 1, inplace = True)
#all_data.drop("characteristic_of_ppp", axis = 1, inplace = True)
#all_data.drop("hour", axis = 1, inplace = True)
#all_data.drop("latitude", axis = 1, inplace = True)


# In[17]:

all_data.columns


# In[18]:

all_data_na_again = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na_again = all_data_na_again.drop(all_data_na_again[all_data_na_again == 0].index).sort_values(ascending=False)[:30]
missing_val = pd.DataFrame({'Missing Ratio' :all_data_na_again})
missing_val


# In[19]:

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na_again.index, y=all_data_na_again)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()


# In[20]:

missing_val.index


# In[21]:

#Since the features must be following a similar trend for a particular day
#of a particular month, filling of NaN values is filled by taking the 
#median of the feature values correspong to that month and day.
for col in missing_val.index:
    all_data[col] = all_data.groupby(["month","day"])[col].transform(lambda x: x.fillna(x.median()))
# all_data["swell_period"] = all_data.groupby(["day"])["swell_period"].transform(lambda x: x.fillna(x.median()))
# all_data["swell_height"] = all_data.groupby(["day"])["swell_height"].transform(lambda x: x.fillna(x.median()))
# all_data["swell_direction"] = all_data.groupby(["day"])["swell_direction"].transform(lambda x: x.fillna(x.median()))
# all_data["dewpoint_temperature"] = all_data.groupby(["day"])["dewpoint_temperature"].transform(lambda x: x.fillna(x.median()))
# all_data["amt_pressure_tend"] = all_data.groupby(["day"])["amt_pressure_tend"].transform(lambda x: x.fillna(x.median()))
# all_data["characteristic_of_ppp"] = all_data.groupby(["day"])["characteristic_of_ppp"].transform(lambda x: x.fillna(x.median()))
# all_data["wave_period"] = all_data.groupby(["day"])["wave_period"].transform(lambda x: x.fillna(x.median()))
# all_data["sea_surface_temp"] = all_data.groupby(["day"])["sea_surface_temp"].transform(lambda x: x.fillna(x.median()))
# all_data["sst_measurement_method"] = all_data.groupby(["day"])["sst_measurement_method"].transform(lambda x: x.fillna(x.median()))
# all_data["air_temperature"] = all_data.groupby(["day"])["air_temperature"].transform(lambda x: x.fillna(x.median()))
# all_data["wind_speed"] = all_data.groupby(["day"])["wind_speed"].transform(lambda x: x.fillna(x.median()))
# all_data["dup_check"] = all_data.groupby(["day"])["dup_check"].transform(lambda x: x.fillna(x.median()))
# all_data["sea_level_pressure"] = all_data.groupby(["day"])["sea_level_pressure"].transform(lambda x: x.fillna(x.median()))


# In[22]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness


# In[23]:

#Separating the columns having skewness value > 0.75.

more_skew_cols = []
less_skew_cols = []
for col in all_data.columns:
    if abs(skewness['Skew'][col]) > 0.75:
        more_skew_cols.append(col)
    else:
        less_skew_cols.append(col)


# In[24]:

skewness = skewness.ix[more_skew_cols]
skewness


# In[25]:

#For each feature having skewness > 0.75, the optimal value of lambda for
#box-cox transform that gives the minimum skewness corresponding to that 
#feature is found here. Lambda values are varied from 0 to 4 in steps of 
#0.01.
lam_list = (np.arange(401)*0.01).tolist()
feat_lam_dict = {}
for col in more_skew_cols:
    min_skew = 1000.
    for lam in lam_list:
        temp = skew(boxcox1p(all_data[col], lam))
        if temp < min_skew:
            min_skew = temp
            min_lam = lam
    feat_lam_dict[col] = min_lam
feat_lam_dict # This is the dictionary mapping the feature to the optimal
#lambda value for that feature.


# In[26]:

#Applying box-cox transform on the features having skewness value > 0.75, 
#in order to reduce their skewness.
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

#from scipy.special import boxcox1p
skewed_features = skewness.index

for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], feat_lam_dict[feat])


# In[27]:

train = all_data[:ntrain]
test = all_data[ntrain:]


# In[28]:

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[29]:

#Validation function
#We use the cross_val_score function of Sklearn. 
#However this function has not a shuffle attribut, we add then one line of code, in order to shuffle the dataset prior to cross-validation
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[30]:

#This model may be very sensitive to outliers. So we need to made it more robust on them.
#For that we use the sklearn's Robustscaler() method on pipeline
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# In[31]:

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[32]:

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[33]:

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[34]:

model_xgb.fit(train.values, y_train)
pred = np.expm1(model_xgb.predict(test.values)) - 300


# In[35]:

sub1 = pd.read_csv("sample_sub.csv")


# In[36]:

sub1['wave_height'] = pred


# In[37]:

sub1.to_csv("Submission.csv",index=False)

