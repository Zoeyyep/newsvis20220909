#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 17:57:06 2022

@author: zhaoziyi
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
chunksize = 500000
data=pd.read_csv('/Users/zhaoziyi/Desktop/sma/step1_sample/allnews.csv',chunksize=chunksize)
for i, chuck in enumerate(data):
    chuck.to_csv('df_{}.csv'.format(i)) # i is for chunk number of each iteration
    

#Importing the dataset
"""

df_0 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/df_0.csv")
df_1 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/df_1.csv")
df_2 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/df_2.csv")
df_3 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/df_3.csv")
df_4 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/df_4.csv")
df_5 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/df_5.csv")

#Merge previously split data
df= df_0.append([df_1,df_2,df_3,df_4,df_5])



# Shape of the dataset
print ("The shape of the dataset : ", df.shape)
df.head()

# Dropping the unnecessary columns
df.drop(columns = ['Unnamed: 0','month','day','section'],inplace = True)
df.head()

#Dropping the missing article
index=df[df['article'].isnull()].index
df.drop(index,inplace = True)
df.isnull().sum()#检查有没有空的了

# Distribution of publication years
df['year'].value_counts()

# Distribution of author
df['author'].value_counts()

# Distribution of publication years
df['publication'].value_counts()

# Countplot Publication
plt.rcParams['figure.figsize'] = [30,15]
sns.set(font_scale = 1.2, style = 'whitegrid')
sns_year = sns.countplot(df['publication'], color = 'darkcyan')
plt.xticks(rotation=45)
sns_year.set(xlabel = "Publication", ylabel = "Count", title = "Distribution of Publication")

# Countplot Articles
plt.rcParams['figure.figsize'] = [15, 15]
sns.set(font_scale = 1.2, style = 'whitegrid')
sns_year = sns.countplot(df['year'], color = 'darkcyan')
sns_year.set(xlabel = "Year", ylabel = "Count", title = "Distribution of the articles according to the year")

# Countplot shows the distribution of author
plt.rcParams['figure.figsize'] = [15, 15]
sns.set(font_scale = 1.2, style = 'whitegrid')
df_author = df.author.value_counts().head(100)
sns.barplot(df_author,df_author.index)
sns_year.set(xlabel = "count", ylabel = "author", title = "The most frequency authors")

#Given the overwhelming size of the data set, we take here only articles reported by six news agencies
publish_1 = df.loc[df['publication'] == 'Reuters',['publication','year','title','article']]
publish_2 = df.loc[df['publication'] == 'The New York Times',['publication','year','title','article']]
publish_3 = df.loc[df['publication'] == 'CNBC',['publication','year','title','article']]
publish_4 = df.loc[df['publication'] == 'The Hill',['publication','year','title','article']]
publish_5 = df.loc[df['publication'] == 'People',['publication','year','title','article']]
publish_6 = df.loc[df['publication'] == 'CNN',['publication','year','title','article']]

#导出为excel
publish_1.to_csv('publish_1.csv')
publish_2.to_csv('publish_2.csv')
publish_3.to_csv('publish_3.csv')
publish_4.to_csv('publish_4.csv')
publish_5.to_csv('publish_5.csv')
publish_6.to_csv('publish_6.csv')


publish_1 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/publish_1.csv")
publish_2 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/publish_2.csv")
publish_3 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/publish_3.csv")
publish_4 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/publish_4.csv")
publish_5 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/publish_5.csv")
publish_6 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/publish_6.csv")

publish_1.sample(2000).to_csv('pubsamp_1.csv')
publish_2.sample(2000).to_csv('pubsamp_2.csv')
publish_3.sample(2000).to_csv('pubsamp_3.csv')
publish_4.sample(2000).to_csv('pubsamp_4.csv')
publish_5.sample(2000).to_csv('pubsamp_5.csv')
publish_6.sample(2000).to_csv('pubsamp_6.csv')


pub_1 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/pubsamp_1.csv")
pub_2 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/pubsamp_2.csv")
pub_3 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/pubsamp_3.csv")
pub_4 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/pubsamp_4.csv")
pub_5 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/pubsamp_5.csv")
pub_6 = pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/pubsamp_6.csv")


frames = [pub_1,pub_2,pub_3,pub_4,pub_5,pub_6]
pd.concat(frames).to_csv('allsample.csv')

