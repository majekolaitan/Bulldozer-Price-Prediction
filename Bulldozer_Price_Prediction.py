
## Data Exploration

import gdown
url = 'https://drive.google.com/uc?id=16sH6xjowE3z4pC5s9FDFBV4-iRtfXqY5'
output = 'TrainAndValid.csv'
gdown.download(url, output, quiet=False)

# Import data analysis tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('TrainAndValid.csv', low_memory=False)
df.head()

df.info()

df.columns

df.describe()

fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])

df.SalePrice.plot.hist()

"""## Conversion and Sorting"""

# Make copy of original dataset before making and changes.
df_tmp = df.copy()

# Here datatype of saledate column is object. convert it to datetime datatype
df_tmp['saledate'] = pd.to_datetime(df_tmp['saledate'])
df_tmp['saledate'].dtype

df_tmp.info()

fig, ax = plt.subplots()
ax.scatter(df_tmp["saledate"][:1000], df_tmp["SalePrice"][:1000])

# Sort DataFrame in date order
df_tmp.sort_values(by=["saledate"], inplace=True, ascending=True)
df_tmp.saledate.head(20)

"""## Add datetime parameters for saledate column"""

# Add datetime parameters for saledate
df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayofweek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayofyear"] = df_tmp.saledate.dt.dayofyear

# Drop original saledate
df_tmp.drop("saledate", axis=1, inplace=True)

df_tmp.head().T

"""## Handling missing values ( imputation techniques)"""

# Work on YearMade column where values are equal to 1000.
unique_years = np.sort(df_tmp['YearMade'].unique())
median_year = int(df_tmp['YearMade'].median())
print(unique_years)
print('Median Year is: ', median_year)

df_tmp[df_tmp['YearMade'] == 1000].shape, df_tmp.shape

df_tmp[df_tmp['YearMade'] != 1000]['YearMade'].plot(kind='hist');

plt.hist(df_tmp['YearMade']);

## Replace YearMade with median value

df_tmp['YearMade'].replace(1000, median_year, inplace=True)
np.sort(df_tmp['YearMade'].unique())

"""## Find columns whose type is numeric."""

for col in df_tmp.columns:
  if pd.api.types.is_numeric_dtype(df_tmp[col]):
    if df_tmp[col].isna().sum():
      print(col)

# Fill missing value with median
for col in df_tmp.columns:
  if pd.api.types.is_numeric_dtype(df_tmp[col]):
    if df_tmp[col].isna().sum():
      # add a column which tell if data is missing or not
      df_tmp[col+'_is_missing'] = df_tmp[col].isna()
      df_tmp[col].fillna(df_tmp[col].median(), inplace=True)

# loop doesn't print anything. Hence no numeric column has null value
for col in df_tmp.columns:
  if pd.api.types.is_numeric_dtype(df_tmp[col]):
    if df_tmp[col].isna().sum():
      print(col)

"""## Find and Fill the categorical columns"""

# Find columns whose type is object and convert to category type
for col in df_tmp.columns:
  if pd.api.types.is_object_dtype(df_tmp[col]):
    df_tmp[col] = df_tmp[col].astype('category')

df_tmp.info()

# Now fill the null values.
for col in df_tmp.columns:
  if pd.api.types.is_categorical_dtype(df_tmp[col]):
    df_tmp[col+'_is_missing'] = df_tmp[col].isna()
    df_tmp[col] = df_tmp[col].cat.codes+1

df_tmp.info()

# Check missing values
df_tmp.isna().sum()

df_tmp.head().T

"""
## Splitting data into train/valid sets"""

df_tmp.saleYear.value_counts()

# Split data into training and validation
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

len(df_val), len(df_train)

# Split data into X & y
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape

"""## Building an evaluation function"""

from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_valid, val_preds):
  return np.sqrt(mean_squared_log_error(y_valid, val_preds))


def evaluate_scores(model):
  val_preds = model.predict(X_valid)
  val_preds = np.abs(val_preds)

  model_scores = {
      'R2_Score' : r2_score(y_valid, val_preds),
      'MSLE' : mean_squared_log_error(y_valid, val_preds),
      'RMSLE' : rmsle(y_valid, val_preds),
      'MAE' : mean_absolute_error(y_valid, val_preds),
  }
  return model_scores

"""## Fit Baseline Models


"""

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Model dictionary
baseline_models = {
    'ridge': Ridge(max_iter=2000),
    'lasso': Lasso(),
    'elasticnet': ElasticNet(max_iter=2000),
    'rfr': RandomForestRegressor(max_samples=10000)
}

# Commented out IPython magic to ensure Python compatibility.
# %%time
# baseline_score = {}
# 
# for model_name, model in baseline_models.items():
#   model = model.fit(X_train, y_train)
#   scores = evaluate_scores(model)
#   baseline_score[model_name] = scores

basemodel_df = pd.DataFrame(baseline_score)
basemodel_df

basemodel_df.loc['RMSLE'].plot.bar();

"""## Hyperparameter tuning with RandomizedSearchCV"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# from sklearn.model_selection import RandomizedSearchCV
# 
# # Hyperparameter grid for RandomizedSearchCV
# rfr_rs_grid = {
#     "n_estimators": np.arange(10, 100, 10),
#     "max_depth": [None, 3, 5, 10],
#     "min_samples_split": np.arange(2, 20, 2),
#     "min_samples_leaf": np.arange(1, 20, 2),
#     "max_features": [0.5, 1, "sqrt", "auto"],
#     "max_samples": [10000]
# }
# 
# rfr_rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,random_state=42),
#                                  param_distributions=rfr_rs_grid,
#                                  cv=5,
#                                  n_iter=10)
# rfr_rs_model.fit(X_train, y_train)

rfr_rs_model.best_params_

rfr_rs_model.best_estimator_

# No much improvement
rfr_rs_score = evaluate_scores(rfr_rs_model.best_estimator_)
rfr_rs_df = pd.DataFrame(rfr_rs_score, index=['rfr_rs']).T
rfr_rs_df

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# from sklearn.model_selection import GridSearchCV
# 
# rfr_gs_grid = {
#     "n_estimators": [30, 40, 50],
#     "max_depth": [None, 3],
#     "min_samples_split": [12, 14, 16],
#     "min_samples_leaf": [1,2],
#     "max_features": [0.5, 1],
#     "max_samples": [20000]
# }
# 
# rfr_gs_model = GridSearchCV(RandomForestRegressor(n_jobs=-1,random_state=42),
#                                  param_grid=rfr_gs_grid,
#                                  cv=3)
# rfr_gs_model.fit(X_train, y_train)

rfr_gs_model.best_params_

#  grid search model works better than other two

rfr_gs_score = evaluate_scores(rfr_gs_model.best_estimator_)
rfr_gs_df = pd.DataFrame(rfr_gs_score, index=['rfr_gs']).T
rfr_gs_df

combined_df = pd.concat([rfr_base_df, rfr_rs_df, rfr_gs_df], axis=1)
combined_df

"""## Fitting on Most Ideal Hyperparamters"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# # Most ideal hyperparamters
# rfr_ideal_model = RandomForestRegressor(n_estimators=50,
#                                     min_samples_leaf=1,
#                                     min_samples_split=12,
#                                     max_features=0.5,
#                                     n_jobs=-1,
#                                     max_samples=None,
#                                     random_state=42) # random state so our results are reproducible
# 
# rfr_ideal_model.fit(X_train, y_train)

rfr_ideal_score = evaluate_scores(rfr_ideal_model)
rfr_ideal_df = pd.DataFrame(rfr_ideal_score, index=['rfr_ideal']).T
rfr_ideal_df

"""## Feature Importance"""

feature_importance = rfr_ideal_model.feature_importances_

feature_dict = dict(zip(X_train.columns, feature_importance))

feature_df = pd.DataFrame(feature_dict, index=['Feature Importance']).T
feature_df.sort_values(by='Feature Importance', inplace=True, ascending=False)
feature_df

# YearMade and ProductSize are the two most important features
feature_df[:10].plot.barh();

"""## Prediction on Test data"""

import gdown

url = 'https://drive.google.com/uc?id=1Oe3LskVXQI6l16cas44rB71UMOSNfrl6'
output = 'Test.csv'
gdown.download(url, output, quiet=False)

test_df = pd.read_csv(output)
test_df.head()

# Preprocessing the data (getting the test dataset in the same format as out training dataset)

np.random.seed(42)

def preprocess_df(df_temp):
  df_temp['YearMade'].replace(1000, median_year, inplace=True)

  df_temp['saledate'] = pd.to_datetime(df_temp['saledate'])

  df_temp["saleYear"] = df_temp.saledate.dt.year
  df_temp["saleMonth"] = df_temp.saledate.dt.month
  df_temp["saleDay"] = df_temp.saledate.dt.day
  df_temp["saleDayofweek"] = df_temp.saledate.dt.dayofweek
  df_temp["saleDayofyear"] = df_temp.saledate.dt.dayofyear

  df_temp.drop('saledate', axis=1, inplace=True)

  for col in df_temp.columns:
    if pd.api.types.is_numeric_dtype(df_temp[col]):
      if df_temp[col].isna().sum():
        # add a column which tell if data is missing or not
        df_temp[col+'_is_missing'] = df_temp[col].isna()
        df_temp[col].fillna(df_temp[col].median(), inplace=True)

  for col in df_temp.columns:
    if pd.api.types.is_object_dtype(df_temp[col]):
      df_temp[col] = df_temp[col].astype('category')

  for col in df_temp.columns:
    if pd.api.types.is_categorical_dtype(df_temp[col]):
      df_temp[col+'_is_missing'] = df_temp[col].isna()
      df_temp[col] = df_temp[col].cat.codes+1

  return df_temp

test_temp_df = test_df.copy()

# Apply preprocess function on test data

test_temp_df = preprocess_df(test_temp_df)

# We can find how the columns differ using sets
set(X_train.columns) - set(test_temp_df.columns)

# Add auctioneerID_is_missing to test data because it is missing from it

test_temp_df['auctioneerID_is_missing'] = False

# Make columns of test and train in same order

test_temp_df = test_temp_df.reindex(X_train.columns, axis=1)

assert test_temp_df.columns.tolist() == X_train.columns.tolist(), "Columns are not the same"

# Predict on test data
test_preds = rfr_ideal_model.predict(test_temp_df)

# Create a dataframe to store test_preds
df_preds = pd.DataFrame()
df_preds['SalesID'] = test_df['SalesID']
df_preds['SalePrice'] = test_preds
df_preds


if __name__ == "__main__":
    main()