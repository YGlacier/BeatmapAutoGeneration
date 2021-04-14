import os 
from ConversionFunctions import *

paths = ["../Data/beatmap/train/", "../Data/beatmap/validation/", "../Data/beatmap/test/"]

for path in paths:
    files = os.listdir(path + "original/")
    for file in files:
        beatmap_to_training_data(path + "original/" + file, path + "trimmed/" + file.split(".")[0] + ".dat")