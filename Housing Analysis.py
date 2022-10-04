#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
import urllib.request

DOWNLOAD_ROOT= "https://github.com/ageron/handson-ml/tree/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        print("Downloading...")
    tgz_path = os.path.join(housing_path, "housing.tgz")
    print("Target Path: " + tgz_path)
    target = urllib.request.urlretrieve(housing_url, tgz_path)
    print("Extracting..."+ target[0])
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    print("Download complete")


# In[2]:


import pandas as pd
import numpy as np
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#fetch_housing_data(HOUSING_URL, HOUSING_PATH)
housing = load_housing_data(HOUSING_PATH)
print(housing.head())

print(housing['ocean_proximity'].value_counts())


# In[3]:


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[4]:


housing.head()


# For testing purposes, we'll want a *stratified split* to ensure representative consistency. 
# For example, if you were to conduct a survey of 1000 americans, you wouldn't want to do it purely randomly.
# 51.3% of americans are female for instance, while 48.7 americans are male, so you'd ideally want to have a "strata" of
# 513 females and 487 males in your survey.

# 1. Let's bracket the median income levels into income categories.

# In[5]:


housing["income_cat"]= pd.cut(housing["median_income"], bins=[0.,1.5,3.0,4.5,6.,np.inf], labels=[1,2,3,4,5])
housing["income_cat"].hist()
housing.head()


# Use sklearn to perform the stratified train/test split

# In[6]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# Let's verify if these proportions match the original. 

# In[7]:


strat_test_set["income_cat"].value_counts()/ len(strat_test_set)


# In[8]:


strat_train_set["income_cat"].value_counts()/ len(strat_train_set)


# In[9]:


housing["income_cat"].value_counts()/len(housing)


# Now revert the original dataset to normal, without the new bracketed category

# In[10]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# Let's plot the geographical information of our data set. It should roughly look like California base on our source. 

# In[11]:


housing.plot(kind='scatter', x='longitude', y="latitude")


# In[12]:


housing.plot(kind='scatter', x='longitude', y="latitude", alpha=0.4)


# In[13]:


housing.plot(kind='scatter', x='longitude', y="latitude", alpha=0.4,
            s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True
            )
plt.legend()


# We can check to see how correlated things are with particular values using the .corr() function within Pandas.

# In[14]:


corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# For more correlations at a glance, use scatter matrixes

# In[15]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))


# In[16]:


housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)


# TODO: Add a linear regression line to the above plot

# Consider creating new attributes to find better correlations

# In[17]:


housing["rooms_per_household"]= housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]= housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]= housing["population"]/housing["households"]
housing["income_per_population"]= housing["median_income"]/housing["population"]


# Now rebuild the correlation matrix

# In[18]:


corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# # Data Cleaning: Preparing Data for Machine Learning Algorithms

# ## Missing Data

# First, lets revert to a clean training set by recopying our stratified split training set. Check for missing values per column by combining isnull() with sum().

# In[19]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing.isnull().sum()


# ### One way we can deal with missing values is with fillna, after manually calculating the median

# In[20]:


median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)
median


# ### Another way to deal with missing values is with the Imputer Class from sklearn.

# Imputer Classes can handle missing values in preparation to the creation of training sets. Just set a strategy (e.g. "mean", "median", "most_frequent", "constant") and make it fit. Just be sure you feed it numbers only, so in this example, we'll feed it a "housing_num" data frame instead of the housing data directly.
# 

# In[21]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis = 1)
imputer.fit(housing_num)
imputer.statistics_


# In[22]:


housing_num.median().values


# In[23]:


X = imputer.transform(housing_num)
housing_tr=pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


# ## More on sklearn design

# Here are some core sklearn design principles to remember: 
# #### Consistency
# All objects share a consistent and simple interface of three parts.
#     #### Estimators
#     Any object class (such as SimpleImputer, KNNImputer, MissingIndicator), that can estimate some parameters on a dataset is called an imputer. Estimation is performed by the fit() method, and any other parameter needed to guide the transformation is called a "hyperparameter" (e.g. strategy) and must be set as an instance method).
# 
#     #### Transformers
#     The transform() method takes a dataset in as a parameter and returns a transformed dataset. Such methods may include fit_transform()
# 
#     #### Predictors
#     #### Some estimators (such as cluster classifiers packages like KMeans or MeanShift) contain a predict method that returns a dataset of new predictions upon being given a dataset to work with. These predictors will also have a "score()" method.
#     
# #### Inspection
# All estimator hyperparameters are accessible directly via public instance (e.g. imputer.strategy, KMeans.random_state)
# 
# #### Nonproliferation of classes
# Datasets are represented as NumPy arrays or SciPy sparse matrices instead of homemade classes. Hyperparameters are just regular strings and numbers. 
# 
# #### Composition
# Exisiting building blocks are reused as much as possible to compose other estimators and transformations. For example, one can easily create a Pipeline estimator from an arbritrary sequence of transformer followed by a final estimator. 
# 
# #### Sensible defaults
# Scikit-Learn provides reasonable default values for most parameters for the sake of creating a baseline working system quickly. 

# 

# In[24]:


'''
from sklearn.pipeline import make_pipeline
>>> from sklearn.linear_model import LogisticRegression
pipe = make_pipeline (LogisticRegression)
pipe.fit(housing["median_income"],  housing["median_house_value"])
Pipeline(steps=('logisticregression', LogisticRegression()))
'''


# # Handling Text and Categorical Attributes

# In[34]:


housing[["ocean_proximity"]]


# So we see above how there's a limited set of values, but ML algorithms prefer numbers. One way to deal with this is with the _*Ordinal Encoder*_ module.

# In[35]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing[["ocean_proximity"]])
housing_cat_encoded[:10]


# You can check what categories you have loaded into your encoder using its publically accessable .categories_ reference variable.

# In[36]:


ordinal_encoder.categories_


# Sometimes you'll just want your categories to be hot or cold though. That's when OneHotEncoder might be better.

# In[37]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder= OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing[["ocean_proximity"]])
housing_cat_1hot


# Note the *sparse matrix* output is there to be memory efficient. This can be converted to a more conventional array using the toarray() method. As mentioned with the Sci-Kit Learn design markdown above, you can access the .categories array it internally stores directly.

# In[38]:


housing_cat_1hot.toarray()


# In[39]:


cat_encoder.categories_


# In[ ]:





# In[40]:


housing["total"]


# # Custom Transformers

# Let's create a combined transformer
# 

# In[41]:


from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, population_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix]/X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        


# In[42]:


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
housing_extra_attribs = attr_adder.transform(housing.values)


# # Feature Scaling

# One of the most important transformations you MUST apply is Feature Scaling. You see, Machine Learning Algorithms do not work so well when the numerical attributes feature wildly different scales.
# 
# For example in this case, total number of rooms ranges from 6 to 39,320, while total income ranges from 0 to 16. 
# 
# *BE AWARE* that scaling the *TARGET* values is not required.
# 
# ### Two Common Ways of Scaling
# *min-max scaling* and *standardization*
# 
# #### Min-Max Scaling
# Also known as normalization. Values are shifted and rescaled so they all end up between 0 and 1. Sklearn contains the MinMaxScaler for this purpose. You can also use its *feature_range* hyperparameter if you want something beyond 0 and 1.
# 
# #### Standardization
# First subtracts the mean value, then divides by the standard deviation so that the resulting distribution has a unit variance. Unlike min-max, this does not bound values to a specific range (which could be an issue for some algorithms. for example, neural expect values btw 0 and 1), and is much less susceptible to outliers.
# 
# Sklearn provides the StandardScaler for this.

# In[ ]:





# We can combine transformations using the PIPELINE modules

# In[ ]:





# In[43]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline=Pipeline([('imputer', SimpleImputer(strategy="median")),
                      ('attribs_adder', CombinedAttributesAdder()),
                      ('std_scaler', StandardScaler()),])

housing_num_tr=num_pipeline.fit_transform(housing_num)


# So far, CATEGORICAL and NUMERICAL attributes have been handled seperately. The *'ColumnTransformer'* module from SKLEARN can deal with both at once.

# In[44]:


from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs)])

housing_prepared=full_pipeline.fit_transform(housing)


# In[45]:


TODO - Add commentary from page 71


# # Select and Train a Model

# ## Training and Evaluating on the Training Set

# Thus far we've framed the problem, gathered the data and explored it, sampled a training and test set, and wrote transformation pipelines to clean it up. 
# 
# 

# ### Linear Regression

# In[46]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[47]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))


# Time to check the Root Squared Mean Error for how accurate these predictions were.

# In[49]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# A prediction error of $68,826 is uh, okay? This could indicate model underfitting the training data, or the model isn't powerful enough. 
# 
# Let's try another method now that everything is prepared. Recall that in this situation you can try to select a more powerful model, feed the training algorithm with better features, or reduce the constraints on the model. 
# 
#             Let's begin with a more complex model.

# ### Decision Tree Regressor

# Capable of finding complex nonlinear relationships.

# In[51]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[52]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# With an error of 0.0, either this algorithm is perfect, or it's very severely overfitting the data. 

# # Better Evaluation Using Cross-Validation

# One way to evaluate the Decision Tree model would be to use the *train_test_split()* function.  Split the training and validation set, then train your models against the smaller training set and evaluate them against the validation set. 
# 
# Scikit-Learn's *K-Fold cross-validation* feature can randomly split the training set into X distinct subsets (called *folds*), then train and evaluate the decision tree X times, picking a different fold for evaluation each time and training the other X-1 folds. The result is an array containing X evaluation scores. 
# 
#     Let's take a look, setting our X for cross valuation to 10. 

# In[58]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)


# In[62]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
    print("Root Mean Squared Error:", tree_rmse_scores)

display_scores(tree_rmse_scores)


# Okay. We need to fine tune things. This RMSE is horrible. Let's try RandomForestRegression now.
# 

# In[65]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[66]:


scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-scores)

display_scores(forest_rmse_scores)


# ## Fine Tuning the Model

# While we can go about fiddling with each hyperparameter manually, Sci-Kit Learn once again already has some interesting modules for that.

# ### Grid Search
# Tell Sci-kit learn what hyperparameters you want to experiment with and what values to try out, and it'll use cross-validation to evaluate all possible combinations.

# In[69]:


from sklearn.model_selection import GridSearchCV

param_grid  = [{'n_estimators': [3, 10, 30], 'max_features':[2, 4, 6, 8]},
               {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features':[2,3,4]}]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)


# In[70]:


grid_search.best_params_


# In[71]:


grid_search.best_estimator_


# In[72]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# So in this case, 6 max features and n-estimators of 30 produces the least root mean square error.

# ### Randomized Search
# GridSearch is fine when exploring relatively few combinations, but when the hyperparameter search space is large, RandomizedSearchCV is often preferable.
# 
# It works much like GridSearch, but instead of every combination, it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration. 
# 
# Two advantages:
# 1. If you let the randomized search run for 1,000 iterations, this approach will explore 1,000 different values for each hyperparameter, rather than a few values per hyperparameter with the GridSearch approach.
# 2. Simply by setting the number of iterations, you have more control over the computing budget you want to allocate to hyperparameter search. 

# In[ ]:





# ### Ensemble Methods

# In[74]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[75]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes= num_attribs+extra_attribs+cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# So, maybe attributes like "Near Ocean" or "Island" aren't useful.

# In[83]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[82]:


from scipy import stats
confidence= 0.95
squared_errors = (final_predictions - y_test)**2
np.sqrt(stats.t.interval(confidence, len(squared_errors)-1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))


# In[ ]:




