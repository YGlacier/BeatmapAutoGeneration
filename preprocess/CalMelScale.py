import librosa
import os
import numpy as np

paths = ["../Data/music/train/", "../Data/music/validation/", "../Data/music/test/"]
Ns =[1024, 2048, 4096] 
hop_length = 441

for path in paths:
    files = os.listdir(path + "wav/")
    for file in files:
        if os.path.isfile(path + "wav/" + file):
            y, sr = librosa.load(path + "wav/" + file, sr=44100)
            y, _ = librosa.effects.trim(y)

            for N in Ns:
                amp = np.abs(librosa.stft(y, n_fft=N, hop_length=hop_length))
                mel_filter = librosa.filters.mel(sr=sr, n_fft=N, n_mels=80, fmin=27.5, fmax=16000)
                mel = mel_filter.dot(amp)
                np.savetxt(path + "mel/" + file.split(".")[0] + "_" + str(N) + ".dat", mel)