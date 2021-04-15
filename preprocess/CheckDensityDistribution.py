import os
from ConversionFunctions import *

paths = ["../Data/beatmap/train/", "../Data/beatmap/validation/", "../Data/beatmap/test/"]
density = [0] * 10

for path in paths:
    files = os.listdir(path + "timestamp/")
    for file in files:
        timestamp_file = open(path + "timestamp/" + file)
        lines = timestamp_file.read().splitlines()
        d = float(lines[1])
        if d < 3.0:
            density[0] += 1
        elif d < 3.0 + 1.75 * 1:
            density[1] += 1
        elif d < 3.0 + 1.75 * 2:
            density[2] += 1
        elif d < 3.0 + 1.75 * 3:
            density[3] += 1
        elif d < 3.0 + 1.75 * 4:
            density[4] += 1
        elif d < 3.0 + 1.75 * 5:
            density[5] += 1
        elif d < 3.0 + 1.75 * 6:
            density[6] += 1
        elif d < 3.0 + 1.75 * 7:
            density[7] += 1
        elif d < 3.0 + 1.75 * 8:
            density[8] += 1
        else:
            density[9] += 1

print(density)
# [37, 168, 156, 101, 145, 125, 123, 70, 43, 24]