
import keras_tuner
from tensorflow import keras
import keras
from keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import os

def weighted_binary_cross_entropy(weights: dict, from_logits: bool = False):

    assert 0 in weights
    assert 1 in weights

    def weighted_cross_entropy_fn(y_true, y_pred):
        tf_y_true = tf.cast(y_true, dtype=tf.float64)
        tf_y_pred = tf.cast(y_pred, dtype=tf.float64)

        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])

        ce = K.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
        loss = K.mean(tf.math.multiply(ce, weights_v))

        return loss

    return weighted_cross_entropy_fn

def neuralNetwork(traindata,train_l,valdata,val_l, Model_Selec, options, model=None):

    ''' Function for creating/training the neural network. If not initialized,
    hyperparameter tuning is used to create the architecture. Otherwise,
    The model passed to it is trained further on the new data.
    Inputs:
        <traindata>: The train data
        <train_l>: Training labels
        <valdata>: Validation data
        <val_l>: Validation labels
        <Model_Select>: Whether or not to do hyperparameter tuning
        <loss>: The loss function to use
        <epochs>: Number of epochs to train for
        <weights>: Whether or not to weight the network
        <trainedNetwork>: The trained model, if doing refinement
        <options>: Training options
    Outputs:
        <model>: A trained model
    '''

    # Tune architecture of model.
    if Model_Selec == 1:

        inputSize = traindata.shape[1]

        # Class weights for balancing. 
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_l), y=train_l)
        class_weight_dict = dict(enumerate(class_weights))
        pos = class_weight_dict[1]
        neg = class_weight_dict[0]

        # If running bias initialization
        bias = np.log(pos / neg)

        def build_model(hp):

            # Define the keras model
            model = tf.keras.models.Sequential()

            # Input layer
            model.add(tf.keras.Input(shape=(inputSize,))) 

            # Define number of hidden layers/dropout/batchnumber and associated number of nodes per layer
            for i in range(hp.Int('num_layers', 2, 6)):
                units = hp.Choice('units_' + str(i), values=[30, 36, 30, 41, 45, 60])
                model.add(tf.keras.layers.Dense(units=units, activation='relu'))
                model.add(tf.keras.layers.Dropout(options['Dropout']))
                model.add(tf.keras.layers.BatchNormalization(momentum=options['BatchNorm']))

            # Add final layer
            model.add(tf.keras.layers.Dense(1, activation='softmax'))

            # Select optimizer and learning rate.
            lr = hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])
            optimizer = tf.keras.optimizers.Adam(lr, clipnorm=0.0001)

            # Loss function
            # TODO: Double check that this is the correct focal loss.
            if options['loss'] == "focal":
                loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=options["alpha"], gamma=options["gamma"])
            elif options['loss'] == "weighted":
                loss = weighted_binary_cross_entropy(class_weight_dict)
            else:
                loss = 'categorical_crossentropy'

            # Compilation
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=['accuracy',
                                   tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC()])

            return model

        tuner = keras_tuner.Hyperband(build_model,
                                objective=keras_tuner.Objective('val_auc', direction="max"),
                                max_epochs=options["epochs"],
                                seed=1234,
                                factor=options["factor"],
                                overwrite=True,
                                directory=os.path.normpath('C:/'))


        tuner.search(traindata, train_l, epochs=options["epochs"], validation_data=(valdata, val_l))
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        
    else:

        # Continue Training with new data

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5)

        model.fit(traindata, train_l, validation_data=(valdata, val_l), batch_size=options["batch_size"],
                  epochs=options["epochs"], callbacks=callback, verbose=2)

    return model
