#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import json
import requests
import curl

from os import walk
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import Counter
from wordcloud import WordCloud
from bs4 import BeautifulSoup

from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sns.set(style="white", palette="muted", color_codes=True)


# In[ ]:


import config
username= config.username
password= config.password


# In[3]:





# In[4]:


import requests


# max_records is there so you can get back partial result sets, if you want. Useful for testing.
def get_all_course_data(max_records, category): 
    page_number = 1
    page_size = 100
    category=category
    more_records_exist = True
    all_courses = []

    while more_records_exist:
        api_response = get_courses(page_number, page_size, category)
        results = process_response(api_response)

        # add the new results to our aggregate set
        all_courses += results
    
        # if we get back fewer records than we asked for, we know we reached the end.
        # terminate the loop if we're at the end, or have hit our max.
        if len(results) < page_size or (max_records and len(all_courses) >= max_records):
            more_records_exist = False
        else:
            page_number += 1
            

    return all_courses

def get_courses(page_number, page_size, category): 
    courses_url = "https://www.udemy.com/api-2.0/courses/"

    query_params = {
        "page_size": page_size, 
        "page": page_number ,
        "category":category
    }

    credentials = (username, password)

    return requests.get(url=courses_url, auth=credentials, params=query_params).json()

def process_response(api_response):
    # api_response is a dictionary object which is a representation of the json structure.  
    courses = []

    # The json structure places all of the courses inside the results array
    for course in api_response['results']:
        # we add to the courses list tuples containing id, price, and headline.
        courses.append((course['id'], course['price'], course['headline']))
    return courses

def dataframing(result, category):
    df = pd.DataFrame.from_records(result, columns=['id', 'price', 'headline'])
    df['category']=category
    return df


# In[5]:


get_ipython().run_cell_magic('time', '', "\ncategories=['Design','Marketing','Development','Music','Lifestyle']\n\ndf=pd.DataFrame(columns=['id', 'price', 'headline','category'])\nfor i in range(len(categories)):\n    n = get_all_course_data(max_records=1000, category=categories[i])\n    n = dataframing(n, category=categories[i])\n    df = df.append([n])\n\n    ")


# In[6]:


df.reset_index(inplace=True)


# In[7]:


df.head()
df.drop(['index'],axis=1)


# In[8]:


df.to_csv('data/data_U.csv')


# Problem Statement

# Hypothesis
# 
# 

# ## Data Cleaning

# #### A description  of the dataset
# 

# In[9]:


data=df
data.describe()


# #### data types

# In[10]:


data.dtypes


# In[11]:


data.category = data.category.astype(str)
data.headline = data.headline.astype(str)


# In[18]:


data.price.replace({'Free': 0}, inplace=True)


# In[19]:


data['price'].unique()


# In[21]:


data['price'] = data['price'].str.replace('$', '')


# In[22]:


data.price = data.price.astype(float)
data.id = data.id.astype(int)


# In[32]:


data.loc[data['price'].isnull()]


# In[34]:


data.fillna(0, inplace=True)


# #### columns in the dataset

# In[35]:


data.columns


# In[36]:


#### Convert Subject to Categorical Numbers

#data['subject'].value_counts() 
#data['subject_cat'] = data.subject
#data.subject_cat.replace({'Web Development': 4, 'Business Finance': 3, 'Musical Instruments':2, 'Graphic Design':1 }, inplace=True)
#data.subject_cat.value_counts()


# ## Exploratory Graphs

# In[37]:



plt.figure(figsize= (10,6))
plt.title('Distribution Plot of Course Price', size=20)

sns.distplot(data['price'], bins=15, color='#5e35b1')
plt.show()


# ## Correlations

# ### boxplots

# In[38]:



plt.figure(figsize=(20,10))
ax = sns.boxplot(x="category", y="price", data=data)
ax.set_title('Box Plots of Price Distribution by Subject', size=30)
ax = sns.swarmplot(x="category", y="price", data=data, color='black', alpha=.4)


# In[40]:


ax = sns.boxplot(data.price)
ax = sns.swarmplot(data.price, color='black', alpha=.4)


# ### segement data

# In[41]:


data['price'].describe()


# #### sections for division into 3 categories is the distribution of prices from 0-25%, 25-75% and 75%-100%
# #### sections for division into 2 categories split the price data evenly in half between min and max

# In[42]:


data['price_bracket_3']=0
for i in range(0, len(data)):
    if (data.price[i] <= 20):
        data['price_bracket_3'][i]='1'
    elif (data.price[i] <= 96) and (data.price[i] >= 20):
        data['price_bracket_3'][i]='2'
    else:
        data['price_bracket_3'][i]='3'


# In[43]:


data['price_bracket_2']=0
for i in range(0, len(data)):
    if (data.price[i] <= 100):
        data['price_bracket_2'][i]=0
    else:
        data['price_bracket_2'][i]=1


# #### divided into 3 segements

# In[44]:


data['price_bracket_3'].value_counts()


# #### divided into 2 segments

# In[45]:


data['price_bracket_2'].value_counts()


# In[46]:


l = len(data[data.price_bracket_3==1])
l


# In[47]:


l = len(data[data.price_bracket_3==1])
m = len(data[data.price_bracket_3==2])
h =len(data[data.price_bracket_3==3])


# In[48]:


category_names = ['Low', 'Medium', 'High']
sizes = [l, m, h]
plt.figure(figsize= (2,2), dpi = 227)
figsize=50
plt.pie(sizes, labels=category_names, 
        textprops={'fontsize' :6} , startangle=50,
       autopct= '%1.1f%%')
plt.title('Pie Chart of division of price into 3 brackets, using distribution of prices from 0-25%, 25-75% and 75%-100% of total count ', size=10)
plt.show()


# In[49]:


l = len(data[data.price_bracket_2==0])
h = len(data[data.price_bracket_2==1])


# In[50]:


category_names = ['Low', 'High']
sizes = [l, h]
plt.figure(figsize= (2,2), dpi = 227)
plt.pie(sizes, labels=category_names, 
        textprops={'fontsize' :6} , startangle=50,
       autopct= '%1.1f%%')
plt.title('Pie Chart of division of price into 2 brackets, from 0-100 dollars, and from 101-200 dollars', size=10)

plt.show()


# In[51]:


data.describe()


# In[52]:


## Save as csv


# In[53]:


#data.to_csv('data/cleaned_data.csv')


# # NLP Preprocessing

# In[54]:


#define stopwords


# In[94]:


my_stopwords = ['a','the','and','of','for','by','an', 'to', 'in', 'learn', 'how', 'from', 'with', 'your', 'you', 'course', 'build', 'use', 'using']


# In[95]:


#make function for cleaning and tokenizing


# In[96]:


def clean_message(message,
                 stop_words=set(my_stopwords)):
    
    words = word_tokenize(message.lower())
    
    filtered_words =[]
    
    for word in words:
        if word not in stop_words and word.isalpha():
            filtered_words.append((word))
        
    return filtered_words


# ### NLP Exploration

# In[97]:


data['words'] = 0
total_words =[]
for i in range(len(data['words'])):
    filtered_words = clean_message(data['headline'][i])
    data['words'][i] = filtered_words
    total_words.append(filtered_words)


# In[98]:


data.head(3)


# In[99]:



flat_list = [item for sublist in total_words for item in sublist]

word_list = [''.join(word) for word in flat_list]
as_string = ' '.join(word_list)

unique_words= pd.Series(flat_list).value_counts()
print ('nr of unique words',  unique_words.shape[0])
unique_words.head()


# In[100]:


icon = Image.open('data/thumbs-up.png')
image_mask= Image.new(mode='RGB', size=icon.size, color=(255,255,255))
image_mask.paste(icon, box=icon)
rgb_array=np.array(image_mask) #converts image object to array

word_cloud = WordCloud(mask = rgb_array, background_color='white',
                      max_words=400, colormap='ocean')
word_cloud.generate(as_string)
plt.figure(figsize=[16,8])
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud for words in all price brackets', size=20)
plt.show()


# In[101]:


data_h = data[data.price_bracket_2 == 1]
data_h.reset_index(inplace=True)
data_h['words'] = 0
total_words_h =[]
for i in range(len(data_h['words'])):
    filtered_words = clean_message(data_h['headline'][i])
    data_h['words'][i] = filtered_words
    total_words_h.append(filtered_words)
    
flat_list = [item for sublist in total_words_h for item in sublist]

word_list = [''.join(word) for word in flat_list]
as_string = ' '.join(word_list)

unique_words= pd.Series(flat_list).value_counts()
print ('nr of unique words',  unique_words.shape[0])
unique_words.head()


# In[102]:


icon = Image.open('data/thumbs-up.png')
image_mask= Image.new(mode='RGB', size=icon.size, color=(255,255,255))
image_mask.paste(icon, box=icon)
rgb_array=np.array(image_mask) #converts image object to array

word_cloud = WordCloud(mask = rgb_array, background_color='white',
                      max_words=400, colormap='ocean')
word_cloud.generate(as_string)
plt.figure(figsize=[16,8])
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud for headlines of courses that cost over $100', size=20)
plt.show()


# In[103]:


data_1 = data[data.price_bracket_2 == 0]
data_1.reset_index(inplace=True)
data_1['words'] = 0
total_words_1 =[]
for i in range(len(data_1['words'])):
    filtered_words = clean_message(data_1['headline'][i])
    data_1['words'][i] = filtered_words
    total_words_1.append(filtered_words)
    
flat_list = [item for sublist in total_words_1 for item in sublist]

word_list = [''.join(word) for word in flat_list]
as_string = ' '.join(word_list)

unique_words= pd.Series(flat_list).value_counts()
print ('nr of unique words',  unique_words.shape[0])
unique_words.head()


# In[104]:


icon = Image.open('data/thumbs-up.png')
image_mask= Image.new(mode='RGB', size=icon.size, color=(255,255,255))
image_mask.paste(icon, box=icon)
rgb_array=np.array(image_mask) #converts image object to array

word_cloud = WordCloud(mask = rgb_array, background_color='white',
                      max_words=400, colormap='ocean')
word_cloud.generate(as_string)
plt.figure(figsize=[16,8])
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud for headlines of courses that cost less than $100', size=20)
plt.show()


# ## Generate and Evaluate Predictive Model

# ## model with 2 categories

# In[105]:


vectorizer =  CountVectorizer(stop_words=set(my_stopwords))
all_features = vectorizer.fit_transform(data.headline)

X_train, X_test, y_train, y_test = train_test_split(all_features,
                                                   data.price_bracket_2,
                                                   test_size=0.3,
                                                   random_state=100)

classifier = MultinomialNB()

classifier.fit(X_train,y_train)

y_hat = (classifier.predict(X_test))

nr_corr = (y_test == y_hat).sum()

preds = classifier.predict(X_test)

#and then test the accuracy


number_wrong = sum(abs(y_hat - y_test))
fraction_wrong = (number_wrong/(len(y_test)))




print ("the percent of predictions the model gets wrong is {:.2%}".format(fraction_wrong))
print ("the accuracy of the model is {:.2%}".format(1- fraction_wrong))
print ("the baseline accuracy would be 50%")
print ("the recall is {:.2%}".format(recall_score(y_test, preds)))
print ("the precision is {:.2%}".format(precision_score(y_test, preds)))



# In[106]:


## model with 3 categories


# In[107]:


vectorizer =  CountVectorizer(stop_words=set(my_stopwords))
all_features = vectorizer.fit_transform(data.headline)

X_train, X_test, y_train, y_test = train_test_split(all_features,
                                                   data.price_bracket_3,
                                                   test_size=0.2,
                                                   random_state=100)

classifier = MultinomialNB()

classifier.fit(X_train,y_train)

y_hat = (classifier.predict(X_test))

nr_corr = (y_test == y_hat).sum()

preds = classifier.predict(X_test)

#and then test the accuracy


number_wrong = sum(abs(y_hat - y_test))
fraction_wrong = (number_wrong/(len(y_test)))




print ("the percent of predictions the model gets wrong is {:.2%}".format(fraction_wrong))
print ("accuracy of the model is {:.2%}".format(1- fraction_wrong))
print ("the baseline accuracy would be 33%")

#print ("the recall is {:.2%}".format(recall_score(y_test, preds)))
#print ("the precision is {:.2%}".format(precision_score(y_test, preds)))



# In[ ]:





# In[ ]:




