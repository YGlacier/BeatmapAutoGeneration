import sys
sys.path.append("../")
from model.TimeStampNetwork import TimeStampNetwork

import torch
import torch.optim as optim
import numpy as np
import os

timestamp_data_path = "../Data/beatmap/train/timestamp/"
music_data_path = "../Data/music/train/mel/"

# dim 0: density 0 - 9
# dim 1: per beatmap data
timestamp_list = [[], [], [], [] ,[], [], [], [], [], []]
timestamp_to_music_index = [[], [], [], [] ,[], [], [], [], [], []] # index of corresponding music
# dim 0: per music mel
# dim 1: three different STFT window size data
music_list = []
input_mel_list = []

loaded_music_name = []

timestamp_file_names = os.listdir(timestamp_data_path)

loaded_timestamp_count = 0
for file_path in timestamp_file_names:
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
        input_mel_list.append([])
        for i in range(music_data_1024.shape[1]):
            input_mel_list[-1].append([[],[],[]])
            for j in range(-7,8):
                index = i + j
                if index < 0 or index >= music_data_1024.shape[1]:
                    input_mel_list[-1][-1][0].append([0] * 80)
                    input_mel_list[-1][-1][1].append([0] * 80)
                    input_mel_list[-1][-1][2].append([0] * 80)
                else:
                    input_mel_list[-1][-1][0].append(music_data_1024[:,index])
                    input_mel_list[-1][-1][1].append(music_data_2048[:,index])
                    input_mel_list[-1][-1][2].append(music_data_4096[:,index])
                
    input_mel_list[-1] = np.array(input_mel_list[-1])
    music_index = loaded_music_name.index(music_name)


    # read time stamp data, fill up the blank after the last time stamp
    timestamp_data = np.array([])
    for i in range(2, len(lines)):
        timestamp_data = np.append(timestamp_data, int(lines[i]))
    for i in range(music_list[music_index][0].shape[1] - timestamp_data.shape[0]):
        timestamp_data = np.append(timestamp_data, 0)
    
    timestamp_list[density_categoty].append(timestamp_data)
    timestamp_to_music_index[density_categoty].append(music_index)

    loaded_timestamp_count += 1
    print("Loaded [" + str(loaded_timestamp_count) + "/" + str(len(timestamp_file_names)) + "] : " + file_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
model = TimeStampNetwork()

f_score_list = []
threshold = 0.5

for epoch in range(100):

    model_path = "../Data/model/4/" + str(epoch) + ".model"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    true_positive_count = 0
    false_positive_count = 0
    ground_truth_count = 0

    for density in range(10):
        # output_list = []
        for timestamp_index in range(len(timestamp_list[density])):

            input_mel = input_mel_list[timestamp_to_music_index[density][timestamp_index]]
            true_timestamp = timestamp_list[density][timestamp_index]
            density_vector = np.array([0] * 10)
            density_vector[density] = 1
            density_matrix = np.tile(density_vector, (input_mel.shape[0], 1))

            input_mel = torch.Tensor(input_mel).to(device)
            density_matrix = torch.Tensor(density_matrix).to(device)
            true_timestamp = torch.Tensor(true_timestamp).to(device)

            output = model.forward([input_mel, density_matrix])

            true_timestamp = true_timestamp.to(cpu)
            output = output.to(cpu)

            for i in range(output.shape[0]):
                if true_timestamp[i] == 1:
                    ground_truth_count += 1
            
                if output[i] >= threshold:
                    is_correct = False
                    for j in range(-2, 3):
                        if (i + j) < 0 or (i + j) >= output.shape[0]:
                            continue
                        if true_timestamp[i+j] == 1:
                            is_correct = True
                            continue
                    if is_correct:
                        true_positive_count += 1
                    else:
                        false_positive_count += 1
    if (true_positive_count + false_positive_count) == 0:
        precision = 0
    else:
        precision = float(true_positive_count) / float(true_positive_count + false_positive_count)
    recall = float(true_positive_count) / float(ground_truth_count)

    if np.abs(precision + recall) < 0.001:
        f_score = 0.0
    else:
        f_score = 2 * precision  * recall / (precision + recall)
    f_score_list.append(f_score)

    print("Epoch: " + str(epoch) + ", F-Score = " + str(f_score))

np.savetxt("../Data/model/4/f_scores" + str(threshold) + ".txt", np.array(f_score_list))
