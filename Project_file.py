#!/usr/bin/env python
# coding: utf-8

# <h1><center>Stock Market Trading Bot</center></h1>

# # Importing libraries

# In[1]:


import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.rcParams['animation.html']='jshtml'


# # Loading data

# In[3]:


data=pd.read_csv('HUBCO Share prices/hubco Share Prices.csv',index_col='Date',parse_dates=True)
data.head()


# # Changing data type

# In[4]:


data.Open=data.Open.astype('float')
data.High=data.High.astype('float')
data.Low=data.Low.astype('float')
data.Close=data.Close.astype('float')
data.Volume=data.Volume.astype('float')


# # Dropping irrelevant feature

# In[5]:


data.drop('Symbol',axis=1,inplace=True)


# In[6]:


data.head()


# In[7]:


data.index.dtype


# In[8]:


data['Date']=data.index.astype('str')


# In[9]:


data.info()


# In[10]:


data1=data.iloc[::-1]


# # Basic correlation analysis

# In[10]:


sns.pairplot(data=data)


# # High and low value of shares over time

# In[11]:


plt.figure(figsize=(8,5))
sns.lineplot(data=data[['High','Low']])


# # Reversing Data Frame

# In[12]:


data1=data.iloc[::-1]


# # Backend of Trading Framework

# In[13]:


def shares_trading(X,total_shares,nS1,nS2,nS3,share_prices,model1_pred,model2_pred,model3_pred,low_thres,high_thres):
    if (model1_pred>model2_pred and model1_pred>model3_pred) and (nS1/total_shares)<high_thres:
        if (model2_pred>model3_pred) and (nS3/(total_shares))>low_thres:
            Amount_Gained=((nS3*share_price[2])*.33)
            Brocker_fee=((nS3*share_price[2])*.33)*.15*.01
            nS3-=(nS3*.33)
            X=X+Amount_Gained-Brocker_fee
            Shares_bought=int(Amount_Gained/share_price[0])
            nS1=nS1+Shares_bought
            return nS1,nS2,nS3
        elif (model3_pred>model2_pred) and (nS2/(total_shares))>low_thres:
            Amount_Gained=((nS2*share_price[1])*.33)
            Brocker_fee=((nS2*share_price[1])*.33)*.15*.01
            nS2-=(nS2*.33)
            X=X+Amount_Gained-Brocker_fee
            Shares_bought=Amount_Gained/share_price[0]
            nS1+=Shares_bought
            return nS1,nS2,nS3
        else:
            return nS1,nS2,nS3
        
    if (model2_pred>model1_pred and model2_pred>model3_pred) and (nS2/total_shares)<high_thres:
        if (model1_pred>model3_pred) and (nS3/(total_shares))>low_thres:
            Amount_Gained=((nS3*share_price[2])*.33)
            Brocker_fee=((nS3*share_price[2])*.33)*.15*.01
            nS3-=(nS3*.33)
            X=X+Amount_Gained-Brocker_fee
            Shares_bought=Amount_Gained/share_price[0]
            nS2+=Shares_bought
            return nS1,nS2,nS3
        elif (model3_pred>model1_pred) and (nS1/(total_shares))>low_thres:
            Amount_Gained=((nS1*share_price[0])*.33)
            Brocker_fee=((nS1*share_price[0])*.33)*.15*.01
            nS1-=(nS1*.33)
            X=X+Amount_Gained-Brocker_fee
            Shares_bought=Amount_Gained/share_price[0]
            nS2+=Shares_bought
            return nS1,nS2,nS3
        else:
            return nS1,nS2,nS3
        
    if (model3_pred>model1_pred and model3_pred>model2_pred) and (nS3/total_shares)<high_thres:
        if (model2_pred>model1_pred) and (nS1/(total_shares))>low_thres:
            Amount_Gained=((nS1*share_price[0])*.33)
            Brocker_fee=((nS1*share_price[0])*.33)*.15*.01
            nS1-=(nS1*.33)
            X=X+Amount_Gained-Brocker_fee
            Shares_bought=Amount_Gained/share_price[0]
            nS3+=Shares_bought
            return nS1,nS2,nS3
        elif (model1_pred>model2_pred) and (nS2/(total_shares))>low_thres:
            Amount_Gained=((nS2*share_price[1])*.33)
            Brocker_fee=((nS2*share_price[1])*.33)*.15*.01
            nS2-=(nS2*.33)
            X=X+Amount_Gained-Brocker_fee
            Shares_bought=Amount_Gained/share_price[0]
            nS3+=Shares_bought
            return nS1,nS2,nS3
        else:
            return nS1,nS2,nS3


# # Front End of Trading Framework

# In[12]:


fig=plt.figure(figsize=(10,6))
ax=fig.add_subplot(111)
i=0
x,y=[],[]
while True:
    x.append(data1.Date[i])
    y.append(data1.Close[i])
    ax.plot(x,y)
    plt.xticks(rotation=90)
    ax.legend(['HUBCO SHARES',x[i],y[i]],loc='upper left')
    plt.show()
    fig.canvas.draw()
    time.sleep(1)
    i+=1


# # Initialize basic amount X and number of shares

# In[ ]:


X=100000
nS1=330
nS2=330
nS3=330


# # Unit Testing

# In[ ]:


total_shares=nS1+nS2+nS3
model1_pred=-.12
model2_pred=-.69
model3_pred=-.11
low_thres=.2
high_thres=.5
share_price=[22.5,21.5,12.5]
nS1,nS2,nS3=shares_trading(X,total_shares,nS1,nS2,nS3,share_price,
                           model1_pred,model2_pred,model3_pred,
                           low_thres,high_thres)


# # Investment Portfolio

# In[ ]:


nS1/total_shares,nS2/total_shares,nS3/total_shares


# # Installing Gradient Boosted Tree model

# In[ ]:


get_ipython().system('pip install XGBoost')


# In[ ]:


data=df


# In[ ]:


from sklearn import model_selection
import xgboost as xgb
from sklearn import metrics


# In[ ]:


df['date'] = df.index
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['dayofyear'] = df['date'].dt.dayofyear
df['dayofmonth'] = df['date'].dt.day
df['weekofyear'] = df['date'].dt.weekofyear
    
X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear','Open','High','Low','Volume']]


# In[ ]:


X.head()


# In[ ]:


X.Open=X.Open.astype('float')
X.High=X.High.astype('float')
X.Low=X.Low.astype('float')
X.Volume=X.Volume.astype('float')


# In[ ]:


y=data[['Close']]


# In[ ]:


y=y.astype('float')
X


# In[ ]:


X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)


# # Training model

# In[ ]:


reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train)


# In[ ]:


yhat=reg.predict(X_test)


# In[ ]:


#sns.lineplot(data=yhat)
sns.lineplot(data=y_test)


# In[ ]:


sns.lineplot(data=yhat)


# In[ ]:


metrics.mean_squared_error(y_test,yhat)


# In[ ]:


yhat=reg.predict(X_train)


# In[ ]:


sns.lineplot(data=yhat)


# In[ ]:


sns.lineplot(data=y_test)

