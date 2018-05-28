from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import initializers
from keras.optimizers import RMSprop

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

batch_size = 32
num_classes = 10
epochs = 200
hidden_units = 100
learning_rate = 1e-6
clip_norm = 1.0
print('x_train shape:', x_train.shape)
x_train = x_train.reshape(-1,28,28)
print('x_train shape:', x_train.shape)
x_train = x_train.reshape(-1,28,28)/255
print('x_train shape:', x_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train.shape[1:])
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
model = Sequential()
model.add(SimpleRNN(hidden_units,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=x_train.shape[1:]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
scores = model.evaluate(x_test, y_test, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])