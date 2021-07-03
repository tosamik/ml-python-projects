# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import pandas and matplotlib
import pandas as pd
from matplotlib import pyplot as plt


# %%
# read the file as csv, df stands for dataframe
df = pd.read_csv('winequality-white.csv', sep=';')


# %%
df.columns


# %%
df.describe()


# %%
df.head(5)


# %%
df.isna().sum()


# %%
df.dtypes


# %%
len(df)


# %%
features = df.columns[:-1]
target = df.columns[-1]


# %%
features


# %%
#seperate the dataframe
df_features = df.loc[:,features]
df_target = df.loc[:,[target]]


# %%
df_features.head(5)


# %%
df_target.head(5)


# %%
len(df_features) == len(df_target)


# %%
pea_corr = df_features.corr(method='pearson')


# %%
pea_corr


# %%
import seaborn as sns


# %%
#plot heatmap
plt.figure(figsize=(10,10))
sns.heatmap(pea_corr, annot=True, fmt='.2f')


# %%
plt.figure(figsize=(14,14))
sns.pairplot(df_features)


# %%
from sklearn.model_selection import train_test_split


# %%
#df_dropped_density = df_features.drop(columns=['density'])


# %%
# split data to train and test, default split is 0.25 test size
# X_train, y_train - training set
# X_test, y_test - test set
X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, random_state=20, test_size=.15)


# %%
print(len(X_test), len(X_train), len(df_features))


# %%
# result X_, y_ , this gives shape of the data frame, row and column
X_train.shape


# %%
y_train.shape[0] == X_train.shape[0]


# %%
type(X_train)


# %%
from sklearn.preprocessing import StandardScaler


# %%
# create scaler
scaler = StandardScaler()


# %%
# sets the min/max, sample size, etc based on the training set
# never fit the test set
scaler.fit(X_train)


# %%
# perform the actual scaling
X_train_scaled = scaler.transform(X_train)


# %%
type(X_train_scaled)


# %%
X_train_scaled[:5]


# %%
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import learning_curve


# %%
# create the SGD regression with the appropriate hyper paramters
regression = SGDRegressor(tol=1e-7, max_iter=50000, early_stopping=True)


# %%
# train out model (above) on a small subset of the data, to see if the hyperameters, features, model actually learn
sample_size, train_score, validation_score = learning_curve(regression, X_train_scaled, y_train, cv=3, verbose=1)


# %%
sample_size


# %%
train_score


# %%
validation_score


# %%
# plot training curve
plt.plot(sample_size, train_score.mean(axis=1), label='Train', color='b')
plt.plot(sample_size, validation_score.mean(axis=1), label='validation', color='orange')


# %%
# happy with the learning curve, train on our entire training set
regression = SGDRegressor(tol=1e-7, max_iter=50000, early_stopping=True)


# %%
len(X_train_scaled)


# %%
regression.fit(X_train_scaled, y_train.values.ravel())


# %%
# Scaled test
X_test_scaled = scaler.transform(X_test)


# %%
#predict
y_hat = regression.predict(X_test_scaled)


# %%
y_hat


# %%
y_test.values.ravel()


# %%
from sklearn.metrics import mean_squared_error, r2_score


# %%
mean_squared_error(y_test, y_hat)


# %%
# aim to closer to 100%
r2_score(y_test, y_hat)*100


# %%
plt.figure(figsize=(12,12))
plt.plot(range(len(y_test)), y_test, label="Actual", color='r')
plt.plot(range(len(y_hat)), y_hat, label="Predict", color='b')
plt.grid()
plt.legend()


# %%
from sklearn.linear_model import LinearRegression


# %%
linear = LinearRegression()


# %%
sample_size, train_score, validation_score = learning_curve(linear, X_train_scaled, y_train, cv=3, verbose=1)


# %%
plt.plot(sample_size, train_score.mean(axis=1), label='Train', color='b')
plt.plot(sample_size, validation_score.mean(axis=1), label='validation', color='orange')
plt.legend()
plt.title('Training Curve - Linerar Regression')


# %%
linear = LinearRegression()


# %%
linear.fit(X_train_scaled,y_train.values)


# %%
y_hat = linear.predict(X_test_scaled)


# %%
y_hat


# %%
y_test.values.ravel()


# %%
mean_squared_error(y_test, y_hat)


# %%
r2_score(y_test, y_hat)*100


# %%
from sklearn.preprocessing import PolynomialFeatures


# %%
poly = PolynomialFeatures(degree=2)


# %%
poly.fit(X_train_scaled)


# %%
# fit_transform() to do both fit and trsansform
X_train_scaled_poly = poly.transform(X_train_scaled)


# %%
X_train_scaled.shape


# %%
X_train_scaled_poly.shape


# %%
# use the poly features
sample_size, train_score, validation_score = learning_curve(linear, X_train_scaled_poly, y_train, cv=3, verbose=1)


# %%
plt.plot(sample_size, train_score.mean(axis=1), label='Train', color='b')
plt.plot(sample_size, validation_score.mean(axis=1), label='validation', color='orange')
plt.legend()
plt.title('Training Curve - Linerar Regression with Polynomial Features')


# %%
X_train_scaled_poly = poly.transform(X_train_scaled)


# %%
len(X_train_scaled_poly)


# %%
len(y_train)


# %%
X_test_scaled_poly = poly.transform(X_test_scaled)


# %%
linear = LinearRegression()
linear.fit(X_train_scaled_poly,y_train)


# %%
y_hat = linear.predict(X_test_scaled_poly)


# %%
mean_squared_error(y_test,y_hat)


# %%
r2_score(y_test, y_hat)*100


# %%
linear.coef_


# %%
linear.intercept_

# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=2ab009db-1c9d-40f0-b722-d7845ecc6cd5' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>

