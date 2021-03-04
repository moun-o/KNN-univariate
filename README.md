## KNN K nearest neighbors (univaried case)

To make it short, KNN algorithm is a supervised classfication algorithm who estimates the output based on the K most similar entries which are called nearest neighbours.


## How we compute similarity

similarity usually is computed based on distance between two entries, the distance metrics often are Euclidiean or City-Block.

## implementation
### Part 0 : libraries and parameters
We start by importing pandas and numpy, these 2 libraries are 
```
import pandas as pd
import numpy as np

```
then you fix k, which represent the size of neighbors to elect
```
#choose k, here for example i chosed the first 5 nearest neighbours
k=5
```
### PART 1 : Load data, process it and prepare it

For example here we use Airbnb Data, we have to predict the price based on accommodates


```
#read data
airbnb_list=pd.read_csv('paris_airbnb.csv')
```

after reading the dataset, we have to clean the price column from symbols and convert from string to float


```
#clean the column price, remove comme ',' and '$'
airbnb_list['price']=airbnb_list['price'].str.replace('$','')
airbnb_list['price']=airbnb_list['price'].str.replace(',','')

#convert price column to float
airbnb_list['price']=airbnb_list['price'].astype('float')
```

then we randomize the dataset, it's very important to do this in order to avoid what we call bias in machine learning, so you will divide the data randomly

```
#random the dataset
airbnb_list=airbnb_list.loc[np.random.permutation(len(airbnb_list))]
```
then for example we choose 75% for train, 25% for test

#separate into train data and test data
```
train_data=airbnb_list[0:6000]
test_data=airbnb_list[7000:]
```

### PART 2 : Training implementation

We implement the the predict function which take parameters
train_set: your training dataset
minimum_person: is the minimum person to accomodate
k: the knn parameters 
```
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
```

### PART 3 : Predict the price and exploit model
```
#fix the min person to accommodate
min_pers=3


#here we call the function we implmented in part 2
test_data['predicted_price_base_on_acc'] = test_data['accommodates'].apply(lambda x:predict_price_based_on_min_accom(train_data,min_pers,k))

#display the results
print(test_data[['price','predicted_price_base_on_acc']])
```
### PART 4 : Evaluation
We gonna use 3 metrics, 
MAE: Mean Absolute Error
MSE: Mean Squared Error
RMSE: Root Mean Square Error



```
#MAE is the mean of error ie: sum(|realprice-predicted|)/n
test_data['error']=np.absolute(test_data['price']-test_data['predicted_price_base_on_acc'])
error_MAE=test_data['error'].mean()
print("MAE error is ",error_MAE)

#MSE is the mean of squared error ie: sum(|realprice-predicted|²)/n
test_data['squared_error']=np.absolute((test_data['price']-test_data['predicted_price_base_on_acc'])**2)
error_MSE=test_data['squared_error'].mean()
print("MSE error is ",error_MSE)


#RMSE is the mean of squared error ie: sqrt(sum(|realprice-predicted|²)/n)
print("RMSE error is ",error_MSE**0.5)

```
