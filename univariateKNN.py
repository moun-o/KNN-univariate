#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.options.mode.chained_assignment=None

#PART 0 :-----------> parameters

#choose k, here for example i chosed the first 5 nearest neighbours
k=5



#PART 1 :-----------> data process

#read data
airbnb_list=pd.read_csv('paris_airbnb.csv')

#clean the column price, remove , and $
airbnb_list['price']=airbnb_list['price'].str.replace('$','')
airbnb_list['price']=airbnb_list['price'].str.replace(',','')

#convert price column to float
airbnb_list['price']=airbnb_list['price'].astype('float')

#random the dataset
airbnb_list=airbnb_list.loc[np.random.permutation(len(airbnb_list))]

#separate into train data and test data

train_data=airbnb_list[0:7000]
test_data=airbnb_list[7000:]






# In[2]:


#PART 2 :-----------> Predict the price based on minimum person to accommodate number, implementing function

def predict_price_based_on_min_accom(train_set,minimum_person,k):
    
    #Clone the train dataframe
    copy_list=train_set.copy()
    
    #Similarity based on Manhattan Distance
    copy_list['dist']=copy_list['accommodates'].apply(lambda x: np.abs(x-minimum_person))
    #Sort the list, to get the first k with minimum distance
    copy_list=copy_list.sort_values('dist')
    
    #get the first k nearest neighbours price
    nearest_neigh=copy_list.iloc[0:k]['price']
    
    #get the mean of the k prices
    price=nearest_neigh.mean()
    
    
    return (price)


# In[3]:



#PART 3 :-----------> Execute it 

#lets try for 1 , 4 and 5 persons
print("the price for 1 person accommodation: ",predict_price_based_on_min_accom(airbnb_list,1,k))
print("the price for 4 person accommodation: ",predict_price_based_on_min_accom(airbnb_list,4,k))
print("the price for 5 person accommodation: ",predict_price_based_on_min_accom(airbnb_list,5,k))


# In[4]:


#PART 4 :-----------> Execute it on test set

min_pers=3
test_data['predicted_price_base_on_acc'] = test_data['accommodates'].apply(lambda x:predict_price_based_on_min_accom(train_data,min_pers,k))


# In[5]:


print(test_data[['price','predicted_price_base_on_acc']])


# In[6]:


#PART 5 :-----------> Evaluation

#MAE is the mean of error ie: sum(|realprice-predicted|)/n
test_data['error']=np.absolute(test_data['price']-test_data['predicted_price_base_on_acc'])
error_MAE=test_data['error'].mean()
print("MAE error is ",error_MAE)

#MSE is the mean of squared error ie: sum(|realprice-predicted|²)/n
test_data['squared_error']=np.absolute((test_data['price']-test_data['predicted_price_base_on_acc'])**2)
error_MSE=test_data['squared_error'].mean()
print("MSE error is ",error_MSE)
print("RMSE error is ",error_MSE**0.5)

