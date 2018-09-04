import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from src.FeatureExtraction import FeatureExtraction
from keras.callbacks import EarlyStopping


class Train(object):
    """
    Train deep neural network.
    """
    def __init__(self, datadim = 41, timesteps = 20, numclasses = 10, datafp = ""):
        """
        Initialize model.

        :param datadim: data dimension.
        :param timesteps: number of bins to be fed through for a single sound.
        :param numclasses: total classes.
        :param datafp: file path for data.
        """
        self.extractor = FeatureExtraction()
        self.datafp = datafp
        tf.set_random_seed(0)
        np.random.seed(0)
        self.datadim = datadim
        self.timesteps = timesteps
        self.numclasses = numclasses
        # input data shape: batch size, timesteps, datadim
        self.model = Sequential()

    def trainmodel(self, train, validate, test):
        """
        Train the model given data.

        :param train: training set.
        :param validate: validation set.
        :param test: test set.
        """
        self.model.add(LSTM(256,
                            activation='hard_sigmoid',
                            recurrent_activation='hard_sigmoid',
                            return_sequences=True,
                            input_shape=(self.timesteps, self.datadim)))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.numclasses, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        trainx, trainy = self.extractor.load(train, self.datafp)
        validatex, validatey = self.extractor.load(validate, self.datafp)
        testx, testy = self.extractor.load(test, self.datafp)

        #--------- Add early stopping
        earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

        self.model.fit(trainx,
                       trainy,
                       batch_size=128,
                       epochs=20,
                       callbacks=[earlystop],
                       validation_data=(validatex, validatey))

        #--------- No early stopping.
        # self.model.fit(trainx, trainy, batch_size=128, epochs=epoch,
        #                    validation_data=(validatex, validatey))

        roc, accuracy = self.eval(testx, testy)

    def eval(self, testx, testy):
        """
        Evaluate model.

        :param testx: test data.
        :param testy: test labels.
        :return: ROC and accuracy and shows plot of confusion matrix.
        """
        yprob = self.model.predict_proba(testx, verbose=0)

        ypred = yprob.argmax(axis=-1)
        ytrue = np.argmax(testy, 1)

        roc = roc_auc_score(testy, yprob)
        print("ROC:", round(roc,3))

        score, accuracy = self.model.evaluate(testx, testy, batch_size=32)
        print("\nAccuracy = {:.2f}".format(accuracy))
        p,r,f,s = precision_recall_fscore_support(ytrue, ypred, average='micro')
        print("F-Score:", round(f,2))

        labels = ["airconditioner", "carhorn", "children", "dogbark", "drilling", "engineidle", "gunshot", "jackhammer", "siren", "streetmusic"]
        yprob = self.model.predict_proba(testx, verbose=0)
        ypred = yprob.argmax(axis=-1)
        ytrue = np.argmax(testy, 1)
        cm = confusion_matrix(ytrue, ypred)
        dfcm = pd.DataFrame(cm, labels, labels)
        plt.figure(figsize=(16,8))
        sn.heatmap(dfcm, annot=True, annot_kws={"size": 14}, fmt='g', linewidths=0.5)
        plt.show()

        return roc, accuracy