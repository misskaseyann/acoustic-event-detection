import glob
import librosa
import numpy as np
import os

class FeatureExtraction(object):
    """
    Extract features out of audio data.

    Large portion extracted from:

    http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/
    """
    def __init__(self):
        """
        Initialize with audio file extension and directory holding all data.
        """
        self.ext = "*.wav"
        self.dir = "datasets/UrbanSound8K/test"

    def windows(self, data, size):
        """
        Makes a window for data being vectorized.

        :param data: audio data.
        :param size: the desired total size of the window.
        """
        start = 0
        while start < len(data):
            yield start, start + size
            start += (size / 2)

    def getfeatures(self, subdirs, bands=20, frames=41):
        """
        Feature extraction for audio data.

        :param subdirs: the subdirectories of data to be processed.
        :param bands: number of mfcc bands applied to window of data.
        :param frames: number of frames to reduce data to.
        :return: np array of features and np array of labels.
        """
        window_size = 512 * (frames - 1)
        mfccs = []
        labels = []
        for l, subdir in enumerate(subdirs):
            for fn in glob.glob(os.path.join(self.dir, subdir, self.ext)):
                data, rate = librosa.load(fn)
                label = fn.split('/')[4].split('-')[1][0]
                for (start, end) in self.windows(data, window_size):
                    if (len(data[start:end]) == window_size):
                        signal = data[start:end]
                        mfcc = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=bands).T.flatten()[:, np.newaxis].T
                        mfccs.append(mfcc)
                        labels.append(label)
        features = np.asarray(mfccs).reshape(len(mfccs), bands, frames)
        return np.array(features), np.array(labels, dtype=np.int)

    def encode(self, labels):
        """
        Helper function for saving labels.

        :param labels: numpy array of labels.
        :return encoded labels.
        """
        numlabels = len(labels)
        numunique = len(np.unique(labels))
        encode = np.zeros((numlabels, numunique))
        encode[np.arange(numlabels), labels] = 1
        return encode

    def save(self, fp):
        """
        Save numpy array.

        :param fp: save file path.
        """
        for i in range(1,21):
            fold_name = 'fold' + str(i)
            features, labels = self.getfeatures([fold_name])
            labels = self.encode(labels)
            feature_file = os.path.join(fp, fold_name + '_x.npy')
            labels_file = os.path.join(fp, fold_name + '_y.npy')
            np.save(feature_file, features)
            np.save(labels_file, labels)

    def load(self, folds, fp):
        """
        Load numpy files from folders.
        :param folds: folders being loaded.
        :param fp: file path of folders.
        :return: numpy arrays of features and labels.
        """
        next_fold = False
        for i in range(len(folds)):
            fold_name = 'fold' + str(folds[i])
            feature_file = os.path.join(fp, fold_name + '_x.npy')
            labels_file = os.path.join(fp, fold_name + '_y.npy')
            loaded_features = np.load(feature_file)
            loaded_labels = np.load(labels_file)
            if next_fold:
                features = np.concatenate((features, loaded_features))
                labels = np.concatenate((labels, loaded_labels))
            else:
                features = loaded_features
                labels = loaded_labels
                next_fold = True

        return features, labels

    def mkdir(self, fp):
        """
        Make directory if it doesn't already exist.

        :param fp: file path of directory.
        """
        mydir = os.path.join(os.getcwd(), fp)
        if not os.path.exists(mydir):
            os.makedirs(mydir)
