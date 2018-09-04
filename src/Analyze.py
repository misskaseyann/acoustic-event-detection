import librosa
from librosa import display, core
import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print("Using:",matplotlib.get_backend())
import numpy as np
import os


class Analyze(object):
    """
    Static helper methods for visualizing data.
    """
    @staticmethod
    def load(dir):
        """
        Process audio using librosa.

        :param dir: directory path of sound files.
        :return: sound data and sample rate.
        """
        print("* reading sounds")
        rawsounds = []
        for file in os.listdir(dir):
            x, sr = librosa.load(dir + file)
            rawsounds.append(x)
        print("* finished reading sounds")
        return rawsounds, sr

    @staticmethod
    def plotwave(soundnames, rawsounds):
        """
        Plot the wave of sound data.

        :param soundnames: labels for plotting.
        :param rawsounds: raw audio data for plotting.
        """
        plt.figure(figsize=(25,60))
        i = 1
        for n, f in zip(soundnames, rawsounds):
            plt.subplot(7, 1, i)
            display.waveplot(np.array(f), sr=22050)
            plt.title(n.title())
            i += 1
        plt.suptitle("Waveplots", x=0.5, y=0.915, fontsize=18)
        plt.show()

    @staticmethod
    def plotspectrogram(soundnames, rawsounds):
        """
        Plot the spectrogram of sound data.

        :param soundnames: labels for plotting.
        :param rawsounds: raw audio data for plotting.
        """
        plt.figure(figsize=(25,60))
        i = 1
        for n, f in zip(soundnames, rawsounds):
            plt.subplot(10, 1, i)
            plt.specgram(np.array(f), Fs=22050)
            plt.title(n.title())
            i += 1
        plt.suptitle("Spectrogram", x=0.5, y=0.915, fontsize=18)
        plt.show()

    @staticmethod
    def plotlogspectrogram(soundnames, rawsounds):
        """
        Plot the log spectrogram of sound data.

        :param soundnames: labels for plotting.
        :param rawsounds: raw audio data for plotting.
        """
        plt.figure(figsize=(25,60))
        i = 1
        for n, f in zip(soundnames, rawsounds):
            plt.subplot(7, 1, i)
            d = core.power_to_db(np.abs(librosa.stft(f)) ** 2)
            librosa.display.specshow(d, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title(n.title())
            i += 1
        plt.suptitle("Log power spectrogram", x=0.5, y=0.915, fontsize=18)
        plt.show()

    @staticmethod
    def plotmelspectrogram(soundnames, rawsounds, sr):
        """
        Plot the mel spectrogram of sound data.

        :param soundnames: labels for plotting.
        :param rawsounds: raw audio data for plotting.
        :param sr: sample rate of audio data for plotting.
        """
        plt.figure(figsize=(25,60))
        i = 1
        for n, f in zip(soundnames, rawsounds):
            plt.subplot(7, 1, i)
            #d = np.abs(librosa.stft(f))**2
            s = librosa.feature.melspectrogram(y=f, sr=sr, n_mels=128, fmax=8000)
            librosa.display.specshow(librosa.power_to_db(s, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title(n.title())
            i += 1
        plt.suptitle("Mel spectrogram", x=0.5, y=0.915, fontsize=18)
        plt.show()