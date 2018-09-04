from src.Analyze import Analyze
from src.Train import Train
from src.FeatureExtraction import FeatureExtraction

if __name__=="__main__":
    trainer = Train(datafp="data/np-rnn/")
    trainset = [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 17, 20]
    validationset = [9, 18, 19, 16, 8, 11]
    testset = [9, 10]
    trainer.trainmodel(trainset, validationset, testset)

    #extractor.load([1,2], "data/np-rnn/")

    # -----------------------------
    # extractor = FeatureExtraction()
    # savedir = "data/np-rnn2"
    # extractor.mkdir(savedir)
    # extractor.save(savedir)

    # -----------------------------
    sounds, sr = Analyze.load("datasets/UrbanSound8K/test/audio/")
    labels = ["foreground", "background", "synthesis", "synthesis", "synthesis", "synthesis", "synthesis"]
    Analyze.plotwave(labels, sounds)
    Analyze.plotspectrogram(labels, sounds)
    Analyze.plotlogspectrogram(labels, sounds)
    Analyze.plotmelspectrogram(labels, sounds, sr)