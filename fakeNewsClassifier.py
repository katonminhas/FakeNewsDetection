
# Katon Minhas
# Fake News Classification

# Develop model to classify news as fake or real using the ISOT dataset


#%% Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPool2D
from keras.utils import np_utils

#%% Read data

path = "C:/Users/Katon/Documents/JHU/CreatingAIEnabledSystems/ISOT_Dataset/"

titles = np.genfromtxt(path+"Title.csv", delimiter=",")
texts = np.genfromtxt(path+"Text.csv", delimiter=",")
total = np.hstack((titles[:,0:4], texts))


#%%

xTitle = titles[:,0:4]
xText = texts[:,0:4]
xTotal = total[:,0:8]

y = titles[:,4].reshape(-1,1)


N = xTitle.shape[0]

#%% One hot encode

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)


#%% PCA for dimensionality reduction (did not improve performance)
pca = PCA(n_components=8)
xPCA = pca.fit_transform(xTotal)


#%% divide train test

xTrain, xTest, yTrain, yTest = train_test_split(xTotal, y, test_size=0.2)



#%% Build model

model = Sequential()

model.add(Dense(64, input_shape=(8,), activation='relu', name='input'))
model.add(Dense(32, activation='relu', name='hidden1'))
model.add(Dense(10, activation='relu', name='hidden2'))
model.add(Dense(2, activation='softmax', name='output'))

opt = Adam(lr=0.005)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


#%% train

model.fit(xTrain, yTrain, batch_size=64, epochs=50)


#%% test

model.evaluate(xTest, yTest)



