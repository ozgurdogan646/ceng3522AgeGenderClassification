from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import preProcessData
import warnings
warnings.filterwarnings('ignore')

data, images = preProcessData.createData()

#%% Train-Test Split For Age Prediction
y_age = data['age'].values
x_train_age , x_test_age , y_train_age , y_test_age = train_test_split(images,y_age,test_size = 0.3,random_state = 42)

#%% Train-Test Split For Gender Prediction
y_gender = data['gender'].values
x_train_gender , x_test_gender , y_train_gender , y_test_gender = train_test_split(images,y_gender,test_size = 0.3,random_state = 42)

#%% Label encoding for age data
le=LabelEncoder()
y_train_age_cat = le.fit_transform(y_train_age)
y_test_age_cat =  le.fit_transform(y_test_age)

#%% Reshaping age
x_train_age = x_train_age.reshape(12674, 200, 200,1)
x_test_age = x_test_age.reshape(5433, 200, 200,1)

#%% Reshaping age
x_train_gender = x_train_gender.reshape(12674, 200, 200,1)
x_test_gender = x_test_gender.reshape(5433, 200, 200,1)

#%% Training the modelAge
modelAge = Sequential()
# CONVOLUTIONAL LAYER
modelAge.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(200, 200, 1), activation='relu',))
# POOLING LAYER
modelAge.add(MaxPool2D(pool_size=(2, 2)))
# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
modelAge.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
modelAge.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
modelAge.add(Dense(8, activation='softmax'))


modelAge.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#%% Training the modelGender
modelGender = Sequential()
# CONVOLUTIONAL LAYER
modelGender.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(200, 200, 1), activation='relu',))
# POOLING LAYER
modelGender.add(MaxPool2D(pool_size=(2, 2)))
# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
modelGender.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
modelGender.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
modelGender.add(Dense(3, activation='softmax'))


modelGender.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#%% Get summary of models
modelAge.summary()
modelGender.summary()

#%%Train the model for age
modelAge.fit(x_train_age,y_train_age_cat,epochs=2)

#%% Evaluate modelAge
modelAge.metrics_names
modelAge.evaluate(x_test_age,y_test_age_cat)

#%%Train the model for gender
modelGender.fit(x_train_gender,y_train_gender,epochs=2)

#%% Evaluate modelGender
modelGender.metrics_namess
modelGender.evaluate(x_test_gender,y_test_gender)

#%% Save models
modelAge.save('models/ModelAgePrediction.h5')
modelGender.save('models/ModelGenderPrediction.h5')
#Save encoder for categorilized age
np.save('AgeClasses.npy',le.classes_)




