# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import pandas and matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


# %%
# read the file as csv, df stands for dataframe
df = pd.read_csv('auto-mpg.data')


# %%
# read the first 5 lines from the top
df.head(5)


# %%
headers = ['mpg','cylinders','displacement','horspower','weight','accelaration','model year','origin','car name']
na = ['?']


# %%
# provide our own headers, the delimeters is one or more (+) spaces (\s)
df = pd.read_csv('auto-mpg.data', names=headers, sep='\s+', na_values=na)


# %%
df.head(10)


# %%
df.tail(5)


# %%
#example the data type for each column
df.dtypes


# %%
# description of the data
df.describe()


# %%
# just the mean
df.mean()


# %%
# any null values
df.isna()


# %%
# count the number of nulls in each column
df.isna().sum()


# %%
# % of null in horspower
(6 / len(df))*100


# %%
# report if there is any na for each of the
df.isna().any()


# %%
df.columns


# %%
# get specific column from data frame, column is a series
df.mpg


# %%
# .values => array, the array is not a standard python array, its a numpy array
df.mpg.values


# %%
df.mpg.values.shape


# %%
# for columns with space in the name
df['model year']


# %%
# print out the % of na for each column
# mpg: 0%
# cylinders: 0%
# ...
(df.isna().sum() / len(df))*100


# %%
data_size = len(df)
for c in df.columns :
    na = df[c].isna().sum()
    na_frac = (na / data_size) * 100
    print('%s = %.2f%%'%(c,na_frac))


# %%
df['car name'].unique()


# %%
df['cylinders'].unique()


# %%
# filter - loc, returns the data frame
df.loc[ df.cylinders == 8, ['car name', 'model year'] ] 


# %%
# query/filter
df.loc [df.cylinders > 4]


# %%
# second column after filter is projection
df.loc [ df.cylinders > 4, ['car name']]


# %%
df.loc[ (df.displacement >= 300) & (df.displacement <= 310), ['car name', 'origin']]


# %%
df_fords = df.loc[ df['car name'].str.contains('ford')]
df_fords


# %%
df_fords = df.loc[ df['car name'].str.contains("ford")]
df_fords.sort_values('mpg', ascending=False)


# %%
df.mpg.hist()


# %%
df.weight.plot(kind = 'bar')


# %%
# scatter plot of mpg and cylinders
df.plot(kind='scatter',x='cylinders',y='mpg')


# %%
# create a pie plot on the number of cyliders, use the original data
print('cylinders: ', df.cylinders.unique())
c = df.groupby('cylinders').count()
c.head(5)


# %%
c.mpg.plot(kind='pie')


# %%
y = np.array(c.mpg.values)
plt.pie(y)
plt.show() 


# %%
# import matplotlib
from matplotlib import pyplot as plt


# %%
df.columns


# %%
df.horspower.unique()


# %%
hp = df.horspower
weight = df.weight
plt.plot(range(len(hp)), hp, label = 'HP')
plt.plot(range(len(weight)), weight, label = 'Weight')
plt.legend()


# %%
plt.figure(figsize=(12,12))
plt.plot(range(len(weight)), weight, label='Weight', marker='o', markerfacecolor='r')
plt.legend
plt.grid()
plt.title("My Random Plot")


# %%
len(df.loc[ df['car name'].str.contains('toyota', regex=True)])


# %%
# count of toyota, ford, chrysler
make = [ 'toyota','ford','chrysler' ]
make_count = []
for i in make :
    c = len(df.loc[ df['car name'].str.contains(i, regex=True)])
    make_count.append(c)
make_count


# %%
make_count_1 = [ len(df.loc[ df['car name'].str.contains(i, regex=True)]) for i in make]
make_count_1


# %%
plt.bar(range(len(make)), make_count)
plt.xticks(range(len(make)), make)
plt.xlabel('Make')
plt.ylabel('Count')


# %%
explode = [.1,.1,.1]
plt.pie(make_count, labels=make, autopct="%d", explode=explode)


# %%
df_zerofilled = df.fillna(0)


# %%
df_zerofilled.isna().sum()


# %%
df_fillmean = df.fillna(df.mean())


# %%
df_dropna = df.dropna()


# %%
len(df_dropna)

# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=f5754277-caa3-4647-a25b-f624bd908380' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>

