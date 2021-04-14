import os 
from ConversionFunctions import *

paths = ["../Data/beatmap/train/", "../Data/beatmap/validation/", "../Data/beatmap/test/"]

for path in paths:
    files = os.listdir(path + "trimmed/")
    for file in files:
        trimmed_to_time_stamp(path + "trimmed/" + file, path + "timestamp/" + file)