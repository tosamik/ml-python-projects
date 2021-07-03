# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import json
import requests


# %%
# read the file as csv
df = pd.read_csv('hdb-resale-flat-prices.csv', header = 0)
df = df.sort_values(by = 'month')
target = 'resale_price'


# %%
# add address feature combining block and street name feature
df['address'] = df.block.str.cat(' ' +  df.street_name)
# drop block and street name feature
df.drop(columns = [ 'block', 'street_name', 'flat_type' ], inplace = True)


# %%
# find the longitude and latitude
dict_longitude = {}
dict_latitude = {}
counter = 0
for i in sorted(df.address.unique()) :
    query_string='https://developers.onemap.sg/commonapi/search?searchVal='+str(i)+'&returnGeom=Y&getAddrDetails=Y&pageNum=1'
    resp = requests.get(query_string)
    data = json.loads(resp.content)
    if data['found'] != 0 :
        longitude = data['results'][0]['LONGITUDE']
        latitude = data['results'][0]['LATITUDE']
    else :
        longitude = None
        latitude = None
    dict_longitude[i] = longitude
    dict_latitude[i] = latitude
    print("No: %s, Latitude: %s, Longitude: %s" %(str(counter), latitude, longitude))
    counter += 1
df['longitude'] = df['address'].map(dict_longitude)
df['latitude'] = df['address'].map(dict_latitude)


# %%
# maps town to int
dict_town = {}
counter = 0
for i in sorted(df.town.unique()) :
    dict_town[i] = counter
    counter += 1
df['town'] = df['town'].map(dict_town).astype(np.int64)


# %%
# maps storey_range to int
df['storey_range'] = df['storey_range'].map(lambda x:0.5*int(x[0:2])+0.5*int(x[-2:])).astype(np.int64)


# %%
# map month to int
dict_month = {}
counter = 0
for i in sorted(df['month'].unique()) :
    dict_month[i] = counter
    counter += 1
df['month'] = df['month'].map(dict_month).astype(np.int64)


# %%
# map remaining_lease to int
dict_remaining_lease = {}
counter = 0
for i in sorted(df['remaining_lease'].unique()) :
    dict_remaining_lease[i] = counter
    counter += 1
df['remaining_lease'] = df['remaining_lease'].map(dict_remaining_lease).astype(np.int64)


# %%
# map flat_model to int
dict_flat_model = {}
counter = 0
for i in sorted(df.flat_model.unique()) :
    dict_flat_model[i] = counter
    counter += 1
df['flat_model'] = df['flat_model'].map(dict_flat_model).astype(np.int64)


# %%
# map address to int
dict_address = {}
counter = 0
for i in sorted(df.address.unique()) :
    dict_address[i] = counter
    counter += 1
df['address'] = df['address'].map(dict_address).astype(np.int64)


# %%
# convert float value to int
df['floor_area_sqm'] = df['floor_area_sqm'].astype(np.int64)
df['resale_price'] = df['resale_price'].astype(np.int64)


# %%
df.dtypes


# %%
df.sample(10)


# %%
# extract the features and target from dataframe
X = df.drop(columns = [ target ])
y = df[[target]]


# %%
# find the correlation
pea_corr = X.corr(method = 'pearson')
plt.figure(figsize = (10, 10))
sns.heatmap(pea_corr, annot = True, fmt = '.2f')


# %%
# split data to train and test
# X_train, y_train - training set
# X_test, y_test - test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.13, random_state = 42)


# %%
# verify the X_train and y_train shape
X_train.shape[0] == y_train.shape[0]


# %%
# check the histogram
y_train.hist()


# %%
# define standard scaler, fit and transform X_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


# %%
# Create the SGD Regression with the appropriate hyper parameters
regression = SGDRegressor(tol = 1e-7, max_iter = 50000, early_stopping = True )


# %%
# find train and validation score
sample_size, train_score, validation_score = learning_curve(regression, X_train_scaled, y_train.values.ravel(), cv=3, verbose=1)


# %%
# plot training curve
plt.plot(sample_size, train_score.mean(axis=1), label='Train', color='b')
plt.plot(sample_size, validation_score.mean(axis=1), label='Validation', color='orange')
plt.legend()
plt.title('Training Curve')


# %%
# fit the training data set in SGD regression model
regression = SGDRegressor(tol = 1e-7, max_iter = 50000, early_stopping = True)
regression.fit(X_train_scaled, y_train.values.ravel())


# %%
# find the y_hat
X_test_scaled = scaler.transform(X_test)
y_hat = regression.predict(X_test_scaled)


# %%
# identify the MSE
mean_squared_error(y_test, y_hat)


# %%
# find the r2 score
r2_score(y_test, y_hat) * 100


# %%
# define linear regression and find the train and validation score
linear = LinearRegression()
sample_size, train_score, validation_score = learning_curve(linear, X_train_scaled, y_train, cv=3)


# %%
# plot training curve
plt.plot(sample_size, train_score.mean(axis=1), label='Train', color='b')
plt.plot(sample_size, validation_score.mean(axis=1), label='Validation', color='orange')
plt.legend()
plt.title('Training Curve - LinearRegression')


# %%
linear = LinearRegression()
linear.fit(X_train_scaled, y_train.values)


# %%
# predict using lineare regression model
X_test_scaled = scaler.transform(X_test)
y_hat = linear.predict(X_test_scaled)


# %%
# find the MSE
mean_squared_error(y_test, y_hat)


# %%
# find the r2 score using linear regression model
r2_score(y_test, y_hat) * 100


# %%
# plot the actual vs predict
plt.figure(figsize=(15, 15))
plt.plot(range(len(y_test)), y_test, label="Actual", color='r')
plt.plot(range(len(y_hat)), y_hat, label='Predict', color='b')
plt.grid()
plt.legend()

# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=ce83b2f3-010a-4c80-bf39-fcb0cb08d73e' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>

