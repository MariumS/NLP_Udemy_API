#!/usr/bin/env python
# coding: utf-8

# from IPython.display import HTML
# 
# HTML('''<script>
# code_show=true; 
# function code_toggle() {
#  if (code_show){
#  $('div.input').hide();
#  } else {
#  $('div.input').show();
#  }
#  code_show = !code_show
# } 
# $( document ).ready(code_toggle);
# </script>
# <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')

# In[2]:


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


# In[3]:


sns.set(style="white", palette="muted", color_codes=True)


# In[64]:


#page_size, highest it can go
numb_vals = 100


# In[114]:


get_ipython().run_cell_magic('bash', '', 'curl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Lifestyle&page_size=100\' > data/data_Lifestyle.json\ncurl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Lifestyle&page_size=100&page=10\' > data/data_Lifestyle_2.json\n\n\ncurl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Photography&page_size=100\' > data/data_Photography.json\ncurl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Photography&page_size=100&page=10\' > data/data_Photography_2.json\n\ncurl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Marketing&page_size=100\' > data/data_Marketing.json\ncurl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Marketing&page_size=100&page=10\' > data/data_Marketing_2.json\n\n\ncurl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Development&page_size=100\' > data/data_Development.json\ncurl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Development&page_size=100&page=10\' > data/data_Development_2.json\n\ncurl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Music&page_size=100\' > data/data_Music.json\ncurl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Music&page_size=100&page=10\' > data/data_Music_2.json\n\ncurl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Design&page_size=100\' > data/data_Design.json\ncurl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Design&page_size=100&page=10\' > data/data_Design_2.json\n\n#curl -H "Authorization: Basic UkkyY1N0Y3YxVHFSM3p5dHZYWmJzNTdnNmNtb1NSTE0xWFpvZjZpVDpBSzN2UGFSRVVEMVpnWVZlTG9ZRlJ1Y2s2TEQ4ejZ3Q0xsSVhiMFdid3BGRXcwaGxGbzc2NHdIalJtSENyc2xZUVJPMHdNVWtTc3Jqd0xUT3dvR1FocGhyRmQxWFpiWDFPV3hsdnRFSVJIVUZjRFpMTko1V1FaNHJqWWRuRkU2Rg==" \'https://www.udemy.com/api-2.0/courses/?category=Teaching%20&%20Academics&page_size=600\' > data/data_Teaching.json\n')


# In[116]:


# Opening JSON file 
a = open('data/data_Development.json',) 
a2 = open('data/data_Development_2.json',) 

b= open ('data/data_Marketing.json',)
b2= open ('data/data_Marketing_2.json',)

c=  open ('data/data_Photography.json')
c2=  open ('data/data_Photography_2.json')

d= open ('data/data_Lifestyle.json')
d2= open ('data/data_Lifestyle_2.json')

e= open ('data/data_Music.json')
e2= open ('data/data_Music_2.json')

f= open ('data/data_Design.json')
f2= open ('data/data_Design_2.json')


# returns JSON object as  
# a dictionary 
data_dev = json.load(a) 
data_dev2 = json.load(a2) 

data_mark = json.load(b) 
data_mark2 = json.load(b2) 

data_photo = json.load(c)
data_photo2 = json.load(c2)

data_life = json.load(d)
data_life2 = json.load(d2)

data_music = json.load(e)
data_music2 = json.load(e2)

data_design = json.load(f)
data_design2 = json.load(f2)


# In[117]:


#https://hackersandslackers.com/extract-data-from-complex-json-python/
def extract_values(obj, key):
    """Pull all values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results


# In[118]:


def method1(data):
    for i in range(numb_vals):
        ids = extract_values(data, 'id')
        ids=ids[0:numb_vals]
        price = extract_values(data, 'price')
        headline = extract_values(data, 'headline')
        zippedList =  list(zip(ids, price, headline))
        df = pd.DataFrame(zippedList, columns = ['id','price','headline']) 
        return df


# In[135]:


df4 = method1(data_life)
#df4.append([method1(data_life2)])
#df4['subject']='Lifestyle'


# In[138]:


df4


# In[140]:


df4.append([method1(data_life2)])


# In[142]:


df1 = method1(data_dev)
df1= df1.append([method1(data_dev2)])
df1['subject']='Development'

df2 = method1(data_mark)
df2 = df2.append([method1(data_mark2)])
df2['subject']='Marketing'

df3 = method1(data_photo)
df3 = df3.append([method1(data_photo2)])
df3['subject']='Photography'


df4 = method1(data_life)
df4= df4.append([method1(data_life2)])
df4['subject']='Lifestyle'

df5 = method1(data_music)
df5= df5.append([method1(data_music2)])
df5['subject']='Music'

df6 = method1(data_design)
df6 = df6.append([method1(data_design2)])
df6['subject']='Design'


# In[144]:


df_n = df1.append([df2, df3, df4, df5, df6])


# In[145]:


df_n.reset_index(inplace=True)


# In[146]:


df_n.drop('index', axis=1, inplace=True)


# In[147]:


df_n.to_csv('data/data_n.csv')


# Problem Statement

# Hypothesis
# 
# 

# ## Data Cleaning

# #### A description  of the dataset
# 

# In[148]:


data=df_n
data.describe()


# #### data types

# In[149]:


data.dtypes


# In[150]:


data.subject = data.subject.astype(str)
data.headline = data.headline.astype(str)


# In[151]:


sep = '$'
for i in range(len(data['price'])):
    data['price'][i] = data['price'][i].split(sep, 1)[1]


# In[152]:


data.price = data.price.astype(float)
data.id = data.id.astype(int)


# #### columns in the dataset

# In[153]:


data.columns


# In[154]:


#### Convert Subject to Categorical Numbers


# In[155]:


#data['subject'].value_counts() 


# In[156]:


#data['subject_cat'] = data.subject


# In[157]:


#data.subject_cat.replace({'Web Development': 4, 'Business Finance': 3, 'Musical Instruments':2, 'Graphic Design':1 }, inplace=True)


# In[158]:


#data.subject_cat.value_counts()


# ## Exploratory Graphs

# In[159]:



plt.figure(figsize= (10,6))
plt.title('Distribution Plot of Course Price', size=20)

sns.distplot(data['price'], bins=15, color='#5e35b1')
plt.show()


# ## Correlations

# ### boxplots

# In[160]:



plt.figure(figsize=(20,10))
ax = sns.boxplot(x="subject", y="price", data=data)
ax.set_title('Box Plots of Price Distribution by Subject', size=30)
ax = sns.swarmplot(x="subject", y="price", data=data, color='black', alpha=.4)


# In[161]:


ax = sns.boxplot(data.price)
ax = sns.swarmplot(data.price, color='black', alpha=.4)


# ### segement data

# In[162]:


data['price'].describe()


# #### sections for division into 3 categories is the distribution of prices from 0-25%, 25-75% and 75%-100%
# #### sections for division into 2 categories split the price data evenly in half between min and max

# In[163]:


data['price_bracket_3']=0
for i in range(0, len(data)):
    if (data.price[i] <= 20):
        data['price_bracket_3'][i]='1'
    elif (data.price[i] <= 96) and (data.price[i] >= 20):
        data['price_bracket_3'][i]='2'
    else:
        data['price_bracket_3'][i]='3'


# In[164]:


data['price_bracket_2']=0
for i in range(0, len(data)):
    if (data.price[i] <= 100):
        data['price_bracket_2'][i]=0
    else:
        data['price_bracket_2'][i]=1


# #### divided into 3 segements

# In[165]:


data['price_bracket_3'].value_counts()


# #### divided into 2 segments

# In[166]:


data['price_bracket_2'].value_counts()


# In[167]:


l = len(data[data.price_bracket_3==1])
l


# In[168]:


l = len(data[data.price_bracket_3==1])
m = len(data[data.price_bracket_3==2])
h =len(data[data.price_bracket_3==3])


# In[169]:


category_names = ['Low', 'Medium', 'High']
sizes = [l, m, h]
plt.figure(figsize= (2,2), dpi = 227)
plt.pie(sizes, labels=category_names, 
        textprops={'fontsize' :6} , startangle=50,
       autopct= '%1.1f%%')
plt.title('Pie Chart of division of price into 3 brackets, using distribution of prices from 0-25%, 25-75% and 75%-100% of total count ', size=10)
plt.show()


# In[170]:


l = len(data[data.price_bracket_2==0])
h = len(data[data.price_bracket_2==1])


# In[171]:


category_names = ['Low', 'High']
sizes = [l, h]
plt.figure(figsize= (2,2), dpi = 227)
plt.pie(sizes, labels=category_names, 
        textprops={'fontsize' :6} , startangle=50,
       autopct= '%1.1f%%')
plt.title('Pie Chart of division of price into 2 brackets, from 0-100 dollars, and from 101-200 dollars', size=10)

plt.show()


# In[172]:


data.describe()


# In[173]:


## Save as csv


# In[174]:


#data.to_csv('data/cleaned_data.csv')


# # NLP Preprocessing

# In[175]:


#define stopwords


# In[176]:


my_stopwords = ['a','the','and','of','for','by','an', 'to', 'in']


# In[177]:


#make function for cleaning and tokenizing


# In[178]:


def clean_message(message,
                 stop_words=set(my_stopwords)):
    
    words = word_tokenize(message.lower())
    
    filtered_words =[]
    
    for word in words:
        if word not in stop_words and word.isalpha():
            filtered_words.append((word))
        
    return filtered_words


# ### NLP Exploration

# In[179]:


data['words'] = 0
total_words =[]
for i in range(len(data['words'])):
    filtered_words = clean_message(data['headline'][i])
    data['words'][i] = filtered_words
    total_words.append(filtered_words)


# In[180]:


data.head(3)


# In[181]:



flat_list = [item for sublist in total_words for item in sublist]

word_list = [''.join(word) for word in flat_list]
as_string = ' '.join(word_list)

unique_words= pd.Series(flat_list).value_counts()
print ('nr of unique words',  unique_words.shape[0])
unique_words.head()


# In[182]:


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


# In[183]:


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


# In[184]:


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


# In[185]:


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


# In[186]:


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

# In[188]:


vectorizer =  CountVectorizer(stop_words=set(my_stopwords))
all_features = vectorizer.fit_transform(data.headline)

X_train, X_test, y_train, y_test = train_test_split(all_features,
                                                   data.price_bracket_2,
                                                   test_size=0.3,
                                                   random_state=10)

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
print ("the recall is {:.2%}".format(recall_score(y_test, preds)))
print ("the precision is {:.2%}".format(precision_score(y_test, preds)))



# In[ ]:





# In[ ]:




