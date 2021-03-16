#%%
#Load libraries
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import warnings 
from keras import backend as K
from keras.layers import Input
import os
import random

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

#Define Functions
def make_lists():
    a = []
    b = []
    c = []
    d = []

    xls = pd.ExcelFile('R:/RevenueManagement/Revenue Management/13. Analytics/2. Projects/4. Forecast Model/2021/IndDat (LFL 24+) LS - HD.xlsx')
    df_Txn = pd.read_excel(xls, 'Final_Txn')

    df_AvgChk = pd.read_excel(xls, 'Final_AvgChk')

    df_EvntTrain = pd.read_excel(xls, 'EvntTrain4')

    df_EvntTest = pd.read_excel(xls, 'EvntTest4')


    return df_EvntTrain,df_EvntTest,df_Txn,df_AvgChk


def get_data():
    #get train data
    train_data_path ='R:/RevenueManagement/Revenue Management/13. Analytics/2. Projects/4. Forecast Model/2021/train.csv'
    train = pd.read_csv(train_data_path,dtype={
        'Day': np.string_,
        'WkDay': np.string_,
        'Mnth': np.string_,
        'MWk': np.string_,
        #'YrWk': np.string_,
    }, encoding="ISO-8859-1")

    #get test data
    test_data_path ='R:/RevenueManagement/Revenue Management/13. Analytics/2. Projects/4. Forecast Model/2021/test.csv'
    test = pd.read_csv(test_data_path,dtype={
        'Day': np.string_,
        'WkDay': np.string_,
        'Mnth': np.string_,
        'MWk': np.string_,
        #'YrWk': np.string_,   
    }, encoding="ISO-8859-1")
    return train , test

def get_combined_data():

    target = train.Ind
    train.drop(['Ind'],axis = 1 , inplace = True)
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index'], inplace=True, axis=1)
    return combined, target

def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type : 
        num : to only get numerical columns with no nans
        no_num : to only get nun-numerical columns with no nans
        all : to get any columns with no nans    
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans

def oneHotEncode(df,colNames):
        for col in colNames:
            if( df[col].dtype == np.dtype('object')):
                dummies = pd.get_dummies(df[col],prefix=col)
                df = pd.concat([df,dummies],axis=1)

                #drop the encoded column
                df.drop([col],axis = 1 , inplace=True)
        return df

#Neural Network Calculation
df_EvntTrain, df_EvntTest, df_Txn, df_AvgChk = make_lists()
IntMdNode = 256

FnlOut_Tx = pd.DataFrame()
for i in range(1,56):
    train , test = get_data()

    train['Evnt'] = df_EvntTrain.iloc[:,i]
    test['Evnt'] = df_EvntTest.iloc[:,i]  

    train['Ind'] = df_Txn.iloc[:,i+1]
    #Combine train and test data to process them together
    combined, target = get_combined_data()
    
    #Check for missing values
    num_cols = get_cols_with_no_nans(combined , 'num')
    cat_cols = get_cols_with_no_nans(combined , 'no_num')

    print ('Number of numerical columns with no nan values :',len(num_cols))
    print ('Number of non-numerical columns with no nan values :',len(cat_cols))

    #Hot-one encode
    print('There were {} columns before encoding categorical features'.format(combined.shape[1]))
    combined = oneHotEncode(combined, cat_cols)
    combined = combined.loc[:,~combined.columns.duplicated()]
    print('There are {} columns after encoding categorical features'.format(combined.shape[1]))

    IntNode = combined.shape[1]

    train = combined[:788]
    test = combined[788:]

    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    
    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()

    #Training the model 
    NN_model.fit(train, target, epochs=2000, batch_size=128, validation_split = 0.1, verbose=0)

    predictions = NN_model.predict(test)
    FnlOut_Tx[df_Txn.columns[i+1]] = predictions[:,0]

    K.clear_session()

FnlOut_Tx.to_csv('R:/RevenueManagement/Team Folders/Nilay Doshi/TxFactorsHD.csv',index=False)
print('A submission file has been made')

FnlOut_Ach = pd.DataFrame()
for i in range(1,56):
    train , test = get_data()

    train['Evnt'] = df_EvntTrain.iloc[:,i]
    test['Evnt'] = df_EvntTest.iloc[:,i]  

    train['Ind'] = df_AvgChk.iloc[:,i+1]
    #Combine train and test data to process them together
    combined, target = get_combined_data()
    
    #Check for missing values
    num_cols = get_cols_with_no_nans(combined , 'num')
    cat_cols = get_cols_with_no_nans(combined , 'no_num')

    print ('Number of numerical columns with no nan values :',len(num_cols))
    print ('Number of non-numerical columns with no nan values :',len(cat_cols))

    #Hot-one encode
    print('There were {} columns before encoding categorical features'.format(combined.shape[1]))
    combined = oneHotEncode(combined, cat_cols)
    combined = combined.loc[:,~combined.columns.duplicated()]
    print('There are {} columns after encoding categorical features'.format(combined.shape[1]))

    IntNode = combined.shape[1]

    train = combined[:788]
    test = combined[788:]

    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    
    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()

    #Training the model 
    NN_model.fit(train, target, epochs=2000, batch_size=128, validation_split = 0.1, verbose=0)

    predictions = NN_model.predict(test)
    FnlOut_Ach[df_AvgChk.columns[i+1]] = predictions[:,0]

    K.clear_session()

FnlOut_Ach.to_csv('R:/RevenueManagement/Team Folders/Nilay Doshi/AChFactorsHD.csv',index=False)
print('A submission file has been made')

# %%
