#!/usr/bin/env python
# coding: utf-8

# # Big Data Project-2
# # Notebook-1
# 
# ### Yash Kasundra
# ### ID- a1838670
# 
# ## Covid-19 (cord dataset) create searching similarity tool

# ### Description
# 
# 1. Collect and process pdf data dump from COVID-19 Open Research Dataset Challenge (CORD-19)
# 2.  Analyze the data and provide publication statistics such as the number of publications according to time, location but not limited to. Provide (any type of) visualization for the results.
# 
# 3. Learn sentence embedding from the articles' abstract and main content respectively.
# 
# 4. Build a tool for question answering: given a user input sentence or query, outputs the top 10 most relevant sentences from the data and the source of the data, i.e., the sentence comes from which article.  The tool could be command-line based or a simple Web-based interface. 
# 
# credits: University of Adelaide (4120_COMP_SCI_7209)

# #### Dataset can be Found on Kaggle using this link:  
# 
# https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

# In this notebook we will create a csv file that would contain all the necessary columns for our 2nd notebook.
# Here we will read text data from jason files and then remove all null values to create a dataframe with around 7900 articles data, which we would then use in our 2nd notebook to train our models and do EDA- preprocessing.

# In[1]:


# Importing required libraries

import numpy as np
import pandas as pd
import os, json
import glob
import csv


# Checking Metadata file

# In[2]:


# Reading data directly from dataset hosted on kaggle

df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')


# In[3]:


df.head()


# In[4]:


# Checking all the columns from the in the dataset

df.info()


# In[5]:


# Removing unnecessary columns for our project

df = df.drop(['sha','source_x','doi', 'pmcid' , 'pubmed_id' , 'license' , 'mag_id' , 'who_covidence_id' ,
                          'arxiv_id', 's2_id'],axis = 1)


# In[6]:


# drop rows that have empty pdf_json cells

df.dropna(subset=['pdf_json_files'], inplace=True)


# In[7]:


# create a new df with first 12000 rows

covid_df = df.sample(12000)


# In[8]:


covid_df.info()


# In[9]:


# Dropping rows with any null values

covid_df = covid_df.dropna()


# In[10]:


covid_df.info()


# In[11]:


covid_df["pdf_json_files"] = covid_df["pdf_json_files"].str.split(";").str[0]


# In[12]:


covid_df.head()


# In[13]:


#function to format body_text into block of text 

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    body.encode('utf-8')
    
    return body


# In[14]:


path_to_json ='../input/CORD-19-research-challenge/'

from pandas import DataFrame
body_text = []
for filename in covid_df['pdf_json_files']:
    filename = path_to_json + filename 
    my_json_file = json.load(open(filename, 'r'))
    body_text.append(format_body(my_json_file['body_text']))


# In[15]:


# Creating a column with those data that we just read from jason files

covid_df['body_text'] = body_text


# In[16]:


covid_df.head(10)


# In[17]:


# Saving that dataset into an csv file to be used in our second notebook (to work on jupyter notebook)

covid_df.to_csv('covid_data.csv',index=False)


# In[ ]:




