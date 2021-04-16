import sys
sys.path.append("../")
from model.TimeStampNetwork import TimeStampNetwork

import torch
import numpy as np
import os

model = TimeStampNetwork()

timestamp_data_path = "../Data/beatmap/test/timestamp/"
music_data_path = "../Data/music/test/mel/"

# dim 0: density 0 - 9
# dim 1: per beatmap data
timestamp_list = [[], [], [], [] ,[], [], [], [], [], []]
timestamp_to_music_index = [[], [], [], [] ,[], [], [], [], [], []] # index of corresponding music
# dim 0: per music mel
# dim 1: three different STFT window size data
music_list = []

loaded_music_name = []

timestamp_file_names = os.listdir(timestamp_data_path)

for file_path in timestamp_file_names[0:10]:
    timestamp_file = open(timestamp_data_path + file_path)
    lines = timestamp_file.read().splitlines()
    music_name = lines[0].split(".")[0]

    # get density
    d = float(lines[1])
    if d < 3.0:
        density_categoty = 0
    elif d < 3.0 + 1.75 * 1:
        density_categoty = 1
    elif d < 3.0 + 1.75 * 2:
        density_categoty = 2
    elif d < 3.0 + 1.75 * 3:
        density_categoty = 3
    elif d < 3.0 + 1.75 * 4:
        density_categoty = 4
    elif d < 3.0 + 1.75 * 5:
        density_categoty = 5
    elif d < 3.0 + 1.75 * 6:
        density_categoty = 6
    elif d < 3.0 + 1.75 * 7:
        density_categoty = 7
    elif d < 3.0 + 1.75 * 8:
        density_categoty = 8
    else:
        density_categoty = 9

    # if the music is not in the list, append it to the list and read the music data
    if music_name not in loaded_music_name:
        loaded_music_name.append(music_name)
        music_data_1024 = np.loadtxt(music_data_path + music_name + "_1024.dat")
        music_data_2048 = np.loadtxt(music_data_path + music_name + "_2048.dat")
        music_data_4096 = np.loadtxt(music_data_path + music_name + "_4096.dat")
        music_list.append([music_data_1024, music_data_2048, music_data_4096])
    music_index = loaded_music_name.index(music_name)


    # read time stamp data, fill up the blank after the last time stamp
    timestamp_data = np.array([])
    for i in range(2, len(lines)):
        timestamp_data = np.append(timestamp_data, int(lines[i]))
    for i in range(music_list[music_index][0].shape[1] - timestamp_data.shape[0]):
        timestamp_data = np.append(timestamp_data, 0)
    
    timestamp_list[density_categoty].append(timestamp_data)
    timestamp_to_music_index[density_categoty].append(music_index)

    print("Loaded " + file_path)
    
    
#print(np.array(timestamp_list).shape)
#print(np.array(timestamp_to_music_index).shape)
#print(np.array(music_list).shape)

print(timestamp_to_music_index)
print("")

for i in range(len(timestamp_list)):
    for j in range(len(timestamp_list[i])):
        print(timestamp_list[i][j].shape)
        print(music_list[timestamp_to_music_index[i][j]][0].shape)
        print(music_list[timestamp_to_music_index[i][j]][1].shape)
        print(music_list[timestamp_to_music_index[i][j]][2].shape)
        print("")




'''
data = np.loadtxt("./-Primary-yuiko-in the Garden-------_1024.dat")

print(data[:,15:30].transpose().shape)

a = np.stack([data[:,0:15].transpose(), data[:,15:30].transpose(), data[:,30:45].transpose()])
print(a.shape)
print(model([torch.Tensor([a]), torch.Tensor([[1,0,0,0,0,0,0,0,0,0]])]))
'''