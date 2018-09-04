# LSTM Acoustic Event Classification
Acoustic event detection using recurrent neural networks.

### Introduction

We as humans consider blindness a greater disability than deafness. Our emphasis on vision is evident in the world around us. We have designers, artists, architects, and photographers that have professions built around the concept of visual perception. However, we depend on both vision and hearing to interact with our environment. If the visual input is unclear to us, hearing becomes the center stage of our brains attention. Our senses aren’t experienced individually but rather intertwined with each other in order to create the experiences we perceive in the conscious world.

This emphasis on vision is also true with the research being done in the field of machine learning. Computer vision is attempting to gain high level understanding from extracting information from digital images or videos using convolutional neural networks (CNNs). The success of these neural networks is so great, that it is now classifying objects with an accuracy close to that of humans. However, even computers have a hard time seeing when its only “sense” being utilized is vision. Could these errors be greatly reduced if a combination of computer vision and computer listening were utilized?

<img src="http://i63.tinypic.com/xlh76d.png" width=600/>

### Define Problem

Teaching a computer to “hear” can encompass many different problems. While much of the research is in speech recognition and computational linguistics, there is much more to be done in the field of robust acoustic event classification, detection, and prediction. For this project, I am focusing on utilizing machine learning (ML) for acoustic event classification. The future scope of this research is to lead into acoustic event prediction for failing machinery.

### Dataset

The problem of acoustic event classification is home to a variety of different sub-problems. Due to the lack of quality annotated datasets dedicated to sound events, the focus of this problem was quickly boiled down to a few choices for experimentation. Google within the last year has released a large audio dataset based on YouTube videos, all hand annotated, encompassing 527 classes. This was an enticing dataset but the focus was too broad. I decided to instead use NYU’s Urbansound8K dataset instead. It contains 8,732 labeled sound excerpts lasting up to four seconds of urban sounds. There are ten classes total which are drawn from the urban sound taxonomy: air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, and street music. Each excerpt is taken from field recordings uploaded to the website www.freesound.org. The files are pre-sorted into ten folders but are not optimally balanced. 

### Signal Processing Methods

To get a visual understanding of the different audio samples being processed by the computer, I have three examples of acoustic events. The first is a car horn, next is a siren, and lastly an air conditioner. For each acoustic event, there are four images. The first is the sound wave signal which is the raw wave data of the sample plotted. The second is a spectrogram which is plotted after a fast Fourier transform (FFT) is applied to the signal. The FFT algorithm transforms the sample from its original domain to its representation in the frequency domain. Next is the spectrogram of a logarithmic Fourier transform applied to the signal. Finally, a mel-frequency cepstrum is plotted. This is a representation of the short-term power spectrum of the signal, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. This is important because the mel scale’s frequency bands approximate the human auditory systems response to sound. For reference, this scale is typically used for audio compression.

When looking at the different samples following this page, observe the patterns shown in the frequency spectrums. Each sound has a unique distribution of power in the frequency components. This pattern is a signature that can be found across many samplings. This signature is often used to identify music in apps such as SoundHound or Shazam.

<img src="http://i67.tinypic.com/1z3oweb.png" width=600/>

<img src="http://i63.tinypic.com/2rw0zv8.png" width=600/>

<img src="http://i63.tinypic.com/ajtahf.png" width=600/>

<img src="http://i68.tinypic.com/2ljtl4l.png" width=600/>

### Architecture

After doing some research, it appears that acoustics in ML have (so far) largely been fed through CNNs. This makes feature extraction fairly easy since each sample can be changed into a spectrogram image where the CNN can detect local feature patterns and hierarchical features within different layers of the image. However, acoustic events occur on a time domain which means it is possible to feed time based data into a recurrent neural network (RNN) instead. This type of architecture could allow a model to also predict the noise as it is “listening” to the data being fed through.

<img src="http://i64.tinypic.com/xmlqvo.png" width=400/>

The specific type of RNN that I decided to use is a Long Term Short Memory network (LSTM). These networks are used more commonly with deep learning RNNs which are designed to avoid the long-term dependency problem. This allows the network to learn and remember information for long periods of time which is important for acoustic data. For example, when feeding a dog barking sample into the LSTM network, it will understand the first bark and remember it. Then, as time passes, more bark events will occur and the network will understand that pattern of noise means a barking dog. 

The complete architecture of the program relies on the Python ML framework TensorFlow with the high-level neural network API, Keras, running on top of it. Matplotlib, pandas, and seaborn is used for plotting data. NumPy is used for optimized matrix and vector operations. SciKitLearn is used for training metrics and librosa is used for audio processing.

### Feature Extraction

Most audio recordings are sampled at a rate of 44100 Hz which would mean that if I fed a single second at a time into the network, it would require 44,100 inputs. This is too large and needed to be reduced in order to train at a reasonable amount of time on a laptop. I found a very interesting blog post that had the same problem as myself (see references) which actually took feature extraction methods from the paper Environmental Sound Classification with Convolutional Networks by Karol J. Piczak. Using this as information, I was able to create a method for creating simpler feature vectors.

The first thing to do is split the data up into windows or “chunks” of data. Each chunk of data is then processed using the mel-frequency cepstral coefficient (MFCC) from the librosa library which allows a reduction in the sample rate to approximately 22000 Hz and a total number of MFCCs to be returned. I set the amount to return at 20 to keep the features minimal for easy processing. Each sample of data is run through this processing and then stored in a NumPy array with its proper labels. Some helper methods were made to save and load the arrays when needed. The reason for the use of MFCC is explained in Signal Processing Methods.

### Adjusted Parameters

Layers
I primarily stuck with two LSTM networks and a single densely-connected neural network layer.
Epochs
I started off at 5 epochs but the training would stop at around 3 before overfitting. Once I added more data, I increased the epochs to 20 just to see what would happen and it would typically stop at around 7 epochs.
LSTM Output Dimensionality
This value I would tweak from 100 to 300, but the change was not evident.
Activation Function
The best performing is the tanh function. After trying multiple others, I noticed a huge decrease in accuracy. Using tanh after the first epoch would give an accuracy of approximately 48%. Other functions such as relu and its variations would have a much lower starting accuracy of approximately 15%. Hard sigmoid was the only other promising function with its first epoch showing an accuracy of 38%. After it finished training, the final accuracy was 64%. However, this took twice as long as the tanh function which made it more computationally expensive.
Dropout
Ranging from 0 to 0.5, the best results came from 0.5 dropout. It increased the accuracy by approximately 2%.
Early Stopping
I played with overfitting and early stopping. With overfitting, the model unsurprisingly didn’t work much better than with early stopping. There was about a 3% reduction in accuracy. I did not overfit with the addition of synthesized data. I monitored primarily validation loss and played with the patience. I started to push the patience to 1 epoch to see what it did, which ended up not resulting in anything.

### Performance

With the original dataset, I was able to reach an accuracy of 54% after tweaking the parameters. This had a consistent early stop after two epochs due to too large of a validation loss at a risk of overfitting. I experimented with overfitting the data and allowed the model to train for an additional two epochs which reduced the accuracy to 51%. 

<img src="http://i68.tinypic.com/madljc.png" width=400/>

After experimenting with different parameters, I knew my next step was to either change how many features are represented in the feature vectors or create more data. Out of curiosity, I decided to go down the path of data synthesis. See the section Data Synthesis for more details on that process. I added an additional 10,000 pieces of data which all originated from specific folders of data to make sure they are sounds not being used in validation or testing. The result was a jump in 14% accuracy which is huge. In total the best the model performed was at an accuracy of 68%.

<img src="http://i68.tinypic.com/21b56rk.png" width=400/>

Unfortunately, there is not much out there in the research of acoustic event classification. However, based on the paper Environmental Sound Classification with Convolutional Networks by Karol J. Piczak, his CNN model performed at an accuracy of 73.6% which is not far from the 68% I was able to obtain. As a frame of reference, on a separate dataset of 2000 samples based on environmental sound events in five different classes had an accuracy of untrained human participants at approximately 81%.

<img src="http://i63.tinypic.com/25ri4hu.png" width=400/>

### Difficulties Encountered

The biggest difficulty in this project was my network getting stuck at an accuracy of 54% which, although better than guessing, is not impressive. I tried many different parameter adjustments but they lead to minimal increases in accuracy. To get over this hill, I decided to try a method that was suggested by Andrew Ng in a talk he gave at the Deep Learning School: “Add more data!”

<img src="http://i64.tinypic.com/11lrm94.png" width=400/>

### Data Synthesis

Data augmentation is commonly used with images which means it can also be applied to sound. Thankfully I did not have to build an entire program to perform this work. A library called Scaper by J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J.P. Bello exists for the sole purpose of soundscape synthesis and augmentation. 

<img src="http://i63.tinypic.com/2qdmq09.png" width=400/>

Using this library, I can easily create additional data points by specifying foreground and background events. Using this, Scaper synthesizes new audio data with a range of overlapping event times, number of events, different pitches, and time stretching. I used the events in folder 1 and folder 2 of the Urbansound8k dataset as foreground events. From there I looked at the website www.freesound.org for urban background ambience. I was able to find eight different samples ranging in length from 10 seconds to over 5 minutes. Using Scaper, I then made 10,000 additional data points to add to my training and validation datasets.

As an example of what Scaper does, lets take a foreground sound (dog bark):

<img src="http://i64.tinypic.com/eugen9.png" width=400/>

A background sound (Tel Aviv cityscape):

<img src="http://i63.tinypic.com/xdsea1.png" width=400/>

Combined together with different parameters, we get new data:

<img src="http://i63.tinypic.com/335fq5x.png" width=600/>

### Future Improvements

I would like to try and exceed the accuracy of the research paper by Karol Pczak that covers this dataset. If I am to do that I believe that more data synthesis is the key to the castle. The next approach I would take is to synthesize 10,000 more pieces of data originating from different cityscape background sounds and from folders of foreground sounds not included in the validation or test data. If the first 10,000 data points increased the accuracy by 14%, I would not be surprised that an additional 10,000 would increase the accuracy by another 7% minimum.

<img src="http://i68.tinypic.com/sgn7kw.png" width=400/>

Overall the biggest problem the classifier had was misclassifying machine noises such as drilling, jackhammer, and air conditioner. This makes sense since they are all mechanical noises which all originate from the same class of noises identified in the urban taxonomy. It also had a hard time understanding the differences between children, street music, and dog barks. This might be because the noise is much more random than something mechanical which makes it hard for the LSTM to identify specific patterns. In the end, it would be good to add more original data and synthesized data to see if that would eliminate any of the misclassifications.

Urban Sound Classification – Applying Convolutional Neural Network
	Aaqib Saeed, September 24, 2016
	http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/
Environmental Sound Classification with Convolutional Neural Networks
	Karol J. Pczak, September 17, 2015
	http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf
Nuts and Bolts of Applying Deep Learning
	Andrew Ng, September 24, 2016
	https://www.youtube.com/watch?v=F1ka6a13S9I
Scaper: A Library for Soundscape Synthesis and Augmentation
	J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J.P. Bello, October 2017
	https://github.com/justinsalamon/scaper
FreeSound Database
	https://freesound.org/
