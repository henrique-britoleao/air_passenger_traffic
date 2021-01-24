###############################################################################
# Add python path
###############################################################################
import sys
sys.path.insert(1, '/air_passenger_traffic')

###############################################################################
# Imports
###############################################################################
from sklearn.ensemble import VotingRegressor
from catboost import CatBoostRegressor
from preprocessing.preprocessing import preprocessor
import model.model as model

###############################################################################
# Special import 
'''
Due to problems with the server running the model, we had to import the 
Keras utils used in a specific manner
'''
###############################################################################

from tensorflow import keras
KerasRegressor = keras.wrappers.scikit_learn.KerasRegressor

###############################################################################

def get_estimator():
    '''Returns pipeline with the model to be used on the train data.'''
    # CatBoostRegressor
    boost_reg = CatBoostRegressor(n_estimators = 5000, learning_rate=0.05, 
                                  max_depth=6, verbose=False)
    # add regressor to the pre-precessing pipeline
    pipeline_boost = preprocessor('Boost').steps.append(['model',boost_reg])
    
    # Neural Network
    nn_reg = KerasRegressor(build_fn=model.nn_model, epochs=60, batch_size=16,
                            verbose=False)
    KerasRegressor._estimator_type = "regressor"
    # add regressor to the pre-precessing pipeline
    pipeline_nn = preprocessor('NN').steps.append(['model', nn_reg])
    
    # Voting regressor
    regressor = VotingRegressor(estimators=
        [('boost', pipeline_boost), ('nn', pipeline_nn)]
    )
 
    return regressor
