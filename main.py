import tensorflow as tf
# print('Using tensorflow version : ' , tf.__version__)
from tensorflow import keras
from keras.datasets import mnist

(x_train , y_train) , (x_test , y_test) = mnist.load_data()

'''
print('x_train shape : ' , x_train.shape)
print('y_train shape : ' , y_train.shape)
print('x_test shape : ' , x_test.shape)
print('y_test shape : ' , y_test.shape)
'''

from matplotlib import pyplot as plt

#plot an image eg
'''
plt.imshow(x_train[0] , cmap = 'binary')
plt.show()
'''
#display labels
'''
print(y_train[0])
print(set(y_train))
'''

#one-hot-encoding
from keras.utils import to_categorical
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

'''
print('y_train_encoded shape : ' ,y_train_encoded.shape)
print('y_test_encoded shape : ' ,y_test_encoded.shape)
print(y_train_encoded[0])  # label at 0 index is '5'
'''

#n-dim array to vector
import numpy as np
x_train_reshaped = np.reshape(x_train , (60000 , 784))
x_test_reshaped = np.reshape(x_test ,  (10000 , 784))

'''
print('x_train_reshaped : ' , x_train_reshaped.shape)
print('x_test_reshaped : ' , x_test_reshaped.shape)

print(set(x_train_reshaped[0]))
'''

#data normalization
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)
epsilon = 1e-10
x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

# print(set(x_train_norm[0]))

#creating model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential([
    Dense(128 , activation = 'relu' , input_shape= (784 , )) ,
    Dense(128 , activation = 'relu') ,
    Dense(10 , activation = 'softmax')
])

#compling the model
model.compile(
    optimizer = 'sgd' ,
    loss = 'categorical_crossentropy' ,
    metrics = ['accuracy']
)
#model.summary()

#model training
model.fit(x_train_norm , y_train_encoded , epochs = 3)

#model evaluation
loss , accuracy = model.evaluate(x_test_norm , y_test_encoded)

#prediction on test set
preds = model.predict(x_test_norm)
#print('Shape of preds : ' , preds.shape)

#plotting result
plt.figure(figsize = (12 , 12))
start_index = 0
for i in range(25):
    plt.subplot(5 , 5 , i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    pred = np.argmax(preds[start_index + i])
    gt = y_test[start_index + i]
    col = 'g'
    if pred != gt:
        col = 'r'
    plt.xlabel('i = {} , pred = {} , gt = {}'.format(start_index , pred , gt) , color = col)
    plt.imshow(x_test[start_index + i] , cmap = 'gray')
plt.show()

#prediction 8 is incorrect
plt.plot(preds[8])
plt.show()