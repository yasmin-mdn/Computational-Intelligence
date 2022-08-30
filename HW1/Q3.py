# Q3_graded
# Do not change the above line.
import tensorflow as tf
import random
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from sklearn.utils import shuffle
from keras import regularizers
from keras.utils.np_utils import to_categorical 
# Type your code here

#Q3_graded
tf.random.set_seed(2022)
random.seed(2022)
np.random.seed(2022)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# Q3_graded
from tensorflow.python.keras.engine.input_layer import Input
model = keras.models.Sequential(
    [
     keras.layers.Input(shape=(32,32,3)),
     keras.layers.Flatten(),
     keras.layers.Dense(units=512, 
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-4),
        activity_regularizer=regularizers.l2(1e-5),
        activation='relu'),
     keras.layers.Dense(units=10, activation='softmax'),
    ]
)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), 
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.1),
              metrics=['accuracy'],
              )
y_train = np.array(y_train)
X_train = np.array(X_train)
y_train, X_train = shuffle(y_train, X_train)
result = model.fit(X_train, 
          y_train, 
          batch_size=512,
          epochs=50, 
          shuffle=True,
          validation_split=0.1
         )
ev=model.evaluate(X_test,y_test)

# Q3_graded
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('loss')
plt.show()
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('accuracy')
plt.show()
preds = model.predict(X_test)
my_preds = preds.argmax(axis=1)
print(my_preds[5])
print(y_test[5])

