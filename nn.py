"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""
# Note: change state, reward, experience replay, learning rate, l2 regularization, add bias 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop,Adam
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def neural_net(num_inputs, params):
    model = Sequential()

    # First layer.
    model.add(Dense(
        params[0], kernel_initializer='lecun_uniform', input_dim = num_inputs,
        use_bias = True, bias_initializer = 'zeros'
    ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Second layer.
    model.add(Dense(params[1], kernel_initializer='lecun_uniform',
        use_bias = True, bias_initializer = 'zeros'
    ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Output layer.
    model.add(Dense(1, kernel_initializer='lecun_uniform',
        use_bias = True, bias_initializer = 'zeros'
    ))
    model.add(Activation('linear'))

    rms = RMSprop()
    adam = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=rms)
    # adam optimizer 10^-3


    return model
