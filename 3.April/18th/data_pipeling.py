# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# importing library
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# importinf data set
dataset = pd.read_csv(r"C:\Users\Deepti Jaiswar\Documents\deepti_nareshit_fullstack_datascience\3.April\Data.csv")

#split data 
X= dataset.iloc[:,:-1].values

y = dataset.iloc[:,3].values

#missing value treatment form transformer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0])

X[:,0] = labelencoder_X.fit_transform(X[:,0])

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

#split data 

from sklearn.model_selection import train_test_split

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,train_size=0.8,random_state=0)
#same thing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=0)
