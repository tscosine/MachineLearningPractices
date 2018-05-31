'''Train a Bidirectional LSTM.'''

from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input,merge,TimeDistributed
from CKdata import getCKdata
from keras.utils import to_categorical
np.random.seed(1337)  # for reproducibility
batch_size = 16
maxlen = 20
hidden = 128
x_train,y_train,x_test_P,y_test_P,x_test_N,y_test_N=getCKdata()
y_train=to_categorical(y_train,2)
print('x_train',x_train.shape)
print('y_train',y_train.shape)
W = (y_train > 0).astype('float')
# this is the placeholder tensor for the input sequences
sequence = Input(shape=(maxlen,), dtype='int32')
# this embedding layer will transform the sequences of integers
# into vectors of size 256
embedded = Embedding(
  10000, 
  output_dim=hidden,
  input_length=maxlen,
  mask_zero=True)(sequence)
# apply forwards LSTM
forwards = LSTM(
  output_dim=hidden,
  return_sequences=True)(embedded)
# apply backwards LSTM
backwards = LSTM(
  output_dim=hidden,
  return_sequences=True,
  go_backwards=True)(embedded)
# concatenate the outputs of the 2 LSTMs
merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
after_dp = Dropout(0.5)(merged)
# TimeDistributed for sequence
# change activation to sigmoid?
dense1=Dense(256)(after_dp)
dense2=Dense(2)(dense1)
# output = TimeDistributed(
#   Dense(
#     output_dim=2,
#     activation='softmax'))(after_dp)
model = Model(input=sequence, output=dense2)
# try using different optimizers and different optimizer configs
# loss=binary_crossentropy, optimizer=rmsprop
model.summary()
model.compile(
  loss='categorical_crossentropy',
  metrics=['accuracy'],
  optimizer='adam',
  sample_weight_mode='temporal')
print('Train...')
model.fit(
  x_train, y_train,
  batch_size=batch_size,
  nb_epoch=10,
  shuffle=True,
  sample_weight=W)