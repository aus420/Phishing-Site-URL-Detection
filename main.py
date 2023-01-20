#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from nltk.tokenize import RegexpTokenizer#regxep tokenizer use to split words from text
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from PIL import Image #for getting image in notebook
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import pickle#use to dump model
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv(r"C:\Users\adas0\Downloads\phishing_site_urls.csv\phishing_site_urls.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


sns.countplot(x=df["Label"],data=df)


# In[8]:


tokenizer=RegexpTokenizer(r'[A-Za-z]+')


# In[9]:


tokenizer.tokenize(df.URL[0])#it will fetch all the words from the first url


# In[10]:


#tokenize all the rows
print('Getting words tokenized...')
t0=time.perf_counter()
df['text_tokenized']=df.URL.map(lambda t:tokenizer.tokenize(t))
t1=time.perf_counter()-t0
print('Time taken',t1,'sec')


# In[11]:


df.sample(5)


# In[12]:


stemmer=SnowballStemmer("english")#it will return the root word


# In[13]:


# Getting all the stemmed words
print('Getting words stemmed ...')
t0= time.perf_counter()
df['text_stemmed'] = df['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[14]:


df.sample(5)


# In[15]:


# Joining all the stemmmed words.
print('Get joiningwords ...')
t0= time.perf_counter()
df['text_sent'] = df['text_stemmed'].map(lambda l: ' '.join(l))
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[16]:


bad_sites=df[df["Label"]=='bad']
good_sites=df[df["Label"]=='good']


# In[17]:


bad_sites.head()


# In[18]:


good_sites.head()


# In[19]:


df.head()


# In[20]:


cv=CountVectorizer()#convert a collection of text documents into a matrix of token counts


# In[21]:


feature = cv.fit_transform(df.text_sent) #transform all text which we tokenize and stemed into sparse matrix


# In[22]:


feature[:5].toarray()#convert sparse matrix to array to print transformed features


# In[23]:


trainX, testX, trainY, testY = train_test_split(feature, df.Label)


# In[24]:


lr = LogisticRegression()
lr.fit(trainX,trainY)


# In[25]:


lr.score(testX,testY)


# In[26]:


Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)


# In[27]:


# creating confusing matrix
print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[28]:


# create mnb object
mnb = MultinomialNB()


# In[29]:


mnb.fit(trainX,trainY)


# In[30]:


mnb.score(testX,testY)


# In[31]:


Scores_ml['MultinomialNB'] = np.round(mnb.score(testX,testY),2)


# In[32]:


print('Training Accuracy :',mnb.score(trainX,trainY))
print('Testing Accuracy :',mnb.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(mnb.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(mnb.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[33]:


# Lets compare the two models and find out which one is best.
acc = pd.DataFrame.from_dict(Scores_ml,orient = 'index',columns=['Accuracy'])
sns.set_style('darkgrid')
sns.barplot(acc.index,acc.Accuracy)


# In[34]:


pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())#Logistic regression is the best fit model,so lets make sklearn pipeline using logistic regression


# In[35]:


trainX, testX, trainY, testY = train_test_split(df.URL, df.Label)


# In[36]:


pipeline_ls.fit(trainX,trainY)


# In[37]:


pipeline_ls.score(testX,testY)


# In[38]:


print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[39]:


pickle.dump(pipeline_ls,open('phishing.pkl','wb'))


# In[40]:


loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)


# In[41]:


#testing with new data
predict_bad=['fazan-pacir.rs/temp/libraries/ipad','tubemoviez.exe']
predict_good=['youtube.com/','retailhellunderground.com/']
loaded_model=pickle.load(open('phishing.pkl','rb'))
result=loaded_model.predict(predict_bad)
result2=loaded_model.predict(predict_good)
print(result)
print("*"*20)
print(result2)

