###############################################################################
# Special import 
'''
Due to problems with the server running the model, we had to import the 
Keras utils used in a specific manner
'''
###############################################################################

from tensorflow import keras

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
RootMeanSquaredError = keras.metrics.RootMeanSquaredError
Adam = keras.optimizers.Adam

def nn_model():
    """Returns a Keras Sequential Neural Network."""
    model = Sequential()
    model.add(Dense(512, kernel_initializer='normal',input_dim = 170, 
                    activation='selu'))
    model.add(Dense(256, kernel_initializer='normal',activation='selu'))
    model.add(Dense(256, kernel_initializer='normal',activation='selu'))
    model.add(Dense(128, kernel_initializer='normal',activation='selu'))
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    model.compile(
        loss='mean_absolute_error', optimizer=Adam(learning_rate=0.0003),
         metrics=['RootMeanSquaredError']
    )
    return model