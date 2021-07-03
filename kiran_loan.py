# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import pandas and matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# %%
# read the file as csv, df stands for dataframe
df = pd.read_csv('kiran_loans.csv')


# %%
df.columns


# %%
# summary
df.describe()


# %%
# examine the head and tail
df.head(5)


# %%
df.dtypes


# %%
# check if there are any null values
df.isna().any()


# %%
# use mean for NAs
df.fillna(df.mean(), inplace=True)


# %%
# per column
# df['log.annual.in'].fillna(method='bfill', inplace=True)


# %%
df.isna().sum()


# %%
df.purpose.unique()


# %%
# encoding
from sklearn.preprocessing import LabelEncoder


# %%
purpose_enc = LabelEncoder()


# %%
# fit the encoder
purpose_enc.fit(df.purpose)


# %%
#df.purpose = purpose_enc.transform(df.purpose)
purpose_enc.transform(df.purpose)


# %%
purpose_enc.inverse_transform([0,1,2,3,4,5,6])


# %%
# dummy variables
pur_dummy = pd.get_dummies(df.purpose)
pur_dummy


# %%
df_dummies = pd.concat([ df, pur_dummy ], axis=1)


# %%
df_dummies.head(5)


# %%
# drop purpose column
df_dummies.drop(columns=[ 'purpose' ], inplace=True)


# %%
df_dummies.head(5)


# %%
df.corr(method='pearson')


# %%
import seaborn as sns


# %%
plt.figure(figsize=(12,12))
sns.heatmap(df.drop(columns=['not.fully.paid']).corr(method='pearson'), annot=True, fmt='.2f')


# %%
sns.pairplot(df.drop(columns=['not.fully.paid']))#


# %%
# split data to train and test set
from sklearn.model_selection import train_test_split


# %%
target = 'not.fully.paid'


# %%
# extract the target from df_dummies
df_target = df_dummies[target]
df_target


# %%
# drop target from dummies
df_feature = df_dummies.drop(columns=[target])
df_feature


# %%
features = df_feature.columns
features


# %%
len(df_feature) == len(df_target)


# %%
X_train, X_test, y_train, y_test = train_test_split(df_feature, df_target, random_state=42)


# %%
print('train sizes: ', len(X_train), len(y_train))
print('test sizes: ', len(X_test), len(y_test))


# %%
y_test


# %%
# imbalance
y_train.hist()


# %%
y_train


# %%
# count the number of 0 in target
len(y_train.loc[y_train == 0])


# %%
# count the number of 1 in the target
len(y_train.loc[y_train == 1])


# %%
6033 + 1150 == len(y_train)


# %%
# undersample - major class size to be similar to the minor class
# 0 is the major class
y_train_0 = y_train.loc[y_train==0].sample(n=2000, replace=False)


# %%
len(y_train_0)


# %%
y_train_0.index
X_train_0 = X_train.loc[y_train_0.index]
X_train_0


# %%
y_train_1 = y_train.loc[y_train == 1]
X_train_1 = X_train.loc[y_train_1.index]
X_train_1


# %%
X_train_features = pd.concat([X_train_0, X_train_1], axis = 0)
X_train_features

y_train_target = pd.concat([y_train_0, y_train_1], axis = 0)
y_train_target.head(5)

y_train_target.tail(5)


# %%
df_train_us = pd.concat([X_train_features, y_train_target], axis = 1)
df_train_us

from sklearn.utils import shuffle
df_train_us = shuffle(df_train_us)
df_train_us = shuffle(df_train_us)

X_train_us_features = df_train_us.loc[:,features]
y_train_us_target = df_train_us.loc[:,[target]]
X_train_us_features
y_train_us_target


# %%
# Over sample
print('0= ', len(y_train[y_train == 0]))
print('1= ', len(y_train[y_train == 1]))


# %%
# sample from y_train == 1
y_train_1_4000 = y_train.loc[y_train == 1].sample(n = 4000, replace=True)
y_train_1_4000


# %%
# extract the new corresponding features
X_train_1_4000 = X_train.loc[y_train_1_4000.index]


# %%
X_train_1_4000.head(5)


# %%
y_train_oversample = pd.concat([ y_train, y_train_1_4000 ], axis=0)


# %%
X_train_oversample = pd.concat([ X_train, X_train_1_4000 ], axis=0)
X_train_oversample


# %%
y_train_oversample.hist()


# %%
len(X_train_oversample) == len(y_train_oversample)


# %%
df_train_oversample = pd.concat([ X_train_oversample, y_train_oversample ], axis=1)
df_train_oversample


# %%
df_train_oversample = shuffle(df_train_oversample)


# %%
df_train_oversample.tail(10)


# %%
df_train_features = df_train_oversample.loc[:, features ]
df_train_target = df_train_oversample.loc[:, [ target] ]


# %%
len(df_train_features) == len(df_train_target)


# %%
df_train_features.head(5)


# %%
df_train_target


# %%
df_train_oversample.isna().sum()


# %%
# scaling
from sklearn.preprocessing import StandardScaler


# %%
# create scaller
scaler = StandardScaler()


# %%
df_train_feature_scaled = scaler.fit_transform(df_train_features)


# %%
df_train_features.head(5).values


# %%
df_train_feature_scaled[:5]


# %%
len(df_train_features.columns)


# %%
from sklearn.decomposition import PCA


# %%
len(df_train_feature_scaled[0])


# %%
# to use PCA, you must scale your features, remove all NAs
pca = PCA(n_components=len(df_train_feature_scaled[0]))


# %%
pca.fit(df_train_feature_scaled)


# %%
pca.explained_variance_ratio_


# %%
sum(pca.explained_variance_ratio_)


# %%
# cumulative variance ratio cvr
cvr = []
var_ratio = pca.explained_variance_ratio_
for i in range(len(var_ratio)) :
    sum_comp = sum(var_ratio[0: i + 1])
    cvr.append(sum_comp)
cvr


# %%
plt.figure(figsize=(10,10))
plt.plot(range ( len(cvr)), cvr, marker='o', markerfacecolor='r')
plt.xticks(range(len(cvr)), range(1,len(cvr) + 1))
plt.grid


# %%
pca = PCA(n_components=16)


# %%
df_train_feature_scaled_pca16 = pca.fit_transform(df_train_feature_scaled)


# %%
len(df_train_feature_scaled[0])


# %%
len(df_train_feature_scaled_pca16[0])


# %%
df_train_feature_scaled[0]


# %%
df_train_feature_scaled_pca16[0]


# %%
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import learning_curve


# %%
classifier = SGDClassifier(max_iter=50000, tol=1e-7, early_stopping=True)


# %%
sample_size, train_score, valid_score = learning_curve(classifier, df_train_feature_scaled_pca16, df_train_target.values.ravel(), verbose=1, cv=3, random_state=42)


# %%
plt.figure(figsize=(10,10))
plt.plot(sample_size, train_score.mean(axis=1), label='Train', color='b')
plt.plot(sample_size,valid_score.mean(axis=1), label='Validation', color='orange')
plt.legend()
plt.title('Learning Curve Logistic Regression - PCA16')


# %%
logit = LogisticRegression(tol=1e-7, max_iter=50000)


# %%
sample_size, train_score, valid_score = learning_curve(logit, df_train_feature_scaled_pca16, df_train_target.values.ravel(), verbose=1, cv=3, random_state=42)


# %%
plt.figure(figsize=(10,10))
plt.plot(sample_size, train_score.mean(axis=1), label='Train', color='b')
plt.plot(sample_size,valid_score.mean(axis=1), label='Validation', color='orange')
plt.legend()
plt.title('Learning Curve Logistic Regression - PCA16')


# %%
sample_size, train_score, valid_score = learning_curve(logit, df_train_feature_scaled, df_train_target.values.ravel(), verbose=1, cv=3, random_state=42)


# %%
plt.figure(figsize=(10,10))
plt.plot(sample_size, train_score.mean(axis=1), label='Train', color='b')
plt.plot(sample_size,valid_score.mean(axis=1), label='Validation', color='orange')
plt.legend()
plt.title('Learning Curve Logistic Regression - Without PCA16')


# %%
logit = LogisticRegression(tol=1e-7, max_iter=50000)


# %%
# fit the model
logit.fit(df_train_feature_scaled, df_train_target.values.ravel())


# %%
# prediction
X_test_scaled = scaler.transform(X_test)
X_test_scaled


# %%
y_hat = logit.predict(X_test_scaled)
y_hat


# %%
y_test.values


# %%
from sklearn.metrics import classification_report


# %%
print(classification_report(y_test, y_hat))


# %%
classifier = SGDClassifier(tol=1e-7, max_iter=50000, early_stopping=True)


# %%
classifier.fit(df_train_feature_scaled, df_train_target)


# %%
y_hat = classifier.predict(X_test_scaled)


# %%
print(classification_report(y_test, y_hat))


# %%
from sklearn.svm import SVC


# %%
svm = SVC()


# %%
svm.fit(df_train_feature_scaled, df_train_target)


# %%
y_hat = svm.predict(X_test_scaled)


# %%
print(classification_report(y_test, y_hat))


# %%
from sklearn.metrics import roc_curve, auc


# %%
prob = logit.decision_function(X_test_scaled)
prob


# %%
dec_svm = svm.decision_function(X_test_scaled)


# %%
fpr, tpr, threshold = roc_curve(y_test, prob)


# %%
fpr_svm, tpr_svm, _ = roc_curve(y_test, dec_svm)


# %%
fpr
tpr


# %%
auc_logit = auc(fpr, tpr)
auc_svm = auc(fpr_svm, tpr_svm)


# %%
print('auc: logit: %.3f, svm: %.3f' %(auc_logit, auc_svm))


# %%
# more AUC is better
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(fpr,tpr,label="Logistic Regression: %.3f" %auc_logit, color='g')
ax.plot(fpr_svm,tpr_svm,label="SVM: %.3f" %auc_svm, color='orange')
ax.plot([0,1],[0,1], color ='r')
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("AUC")
fig.legend()
plt.grid()

# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=eab70691-31f7-454b-933c-c51078cafdd8' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>

