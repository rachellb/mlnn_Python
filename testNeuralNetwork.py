import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pathlib  # Create director if it does not exist (requires python 3.5+)


def testNetwork(traindata, train_l, valdata, val_l, options, model=None):
    """
    This is purely for troubleshooting purposes, and gives me a way to make sure that
    any problems with the model aren't due to the data.
    """

    inputSize = traindata.shape[1]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(inputSize,)))
    """
    model.add(tf.keras.layers.Dense(194, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    
    """
    model.add(tf.keras.layers.Dense(60, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(45, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output layer

    optimizer = tf.keras.optimizers.Adam(0.001, clipnorm=0.0001)

    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=3)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()])

    history = model.fit(traindata, train_l, validation_data=(valdata, val_l), epochs=options["epochs"],
                        batch_size=options["batch_size"], verbose=2)

    # testMethod = tf.keras.Model(model.input, model.layers[-1].output)

    formatDirect = "Images/Results/%s/" % options["dataName"]
    pathlib.Path(formatDirect).mkdir(parents=True, exist_ok=True)

    # Graphing results
    plt.clf()
    plt.cla()
    plt.close()

    auc = plt.figure()
    # plt.ylim(0.40, 0.66)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')

    formatFilename = "AUC_Multilevel_%depochs%dRefine%sBorderPoints%dNeighbors%dLoss%smaxIte%d.png"
    filename = formatFilename % (options["multilevel"], options["epochs"], options["refineMethod"],
                                 options["numBorderPoints"], options["n_neighbors"], options["loss"],
                                 options["max_ite"])

    filename = formatDirect + filename
    plt.savefig(filename)

    plt.clf()
    plt.cla()
    plt.close()

    loss = plt.figure()
    # plt.ylim(0.0, 0.15)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')

    formatFilename = "Loss_Multilevel_%depochs%dRefine%sBorderPoints%dNeighbors%dLoss%smaxIte%d.png"
    filename = formatFilename % (options["multilevel"], options["epochs"], options["refineMethod"],
                                 options["numBorderPoints"], options["n_neighbors"], options["loss"],
                                 options["max_ite"])

    filename = formatDirect + filename
    plt.savefig(filename)

    plt.clf()
    plt.cla()
    plt.close()

    loss = plt.figure()
    # plt.ylim(0.0, 0.15)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')

    formatFilename = "Acc_Multilevel_%depochs%dRefine%sBorderPoints%dNeighbors%dLoss%smaxIte%d.png"
    filename = formatFilename % (options["multilevel"], options["epochs"], options["refineMethod"],
                                 options["numBorderPoints"], options["n_neighbors"], options["loss"],
                                 options["max_ite"])
    filename = formatDirect + filename

    plt.savefig(filename)

    return model
