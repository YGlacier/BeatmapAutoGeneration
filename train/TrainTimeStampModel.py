import sys
sys.path.append("../")
from model.TimeStampNetwork import TimeStampNetwork

import torch
import torch.optim as optim
import numpy as np
import os

# load the training data
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
    
# train the model
learn_rate = 0.001
seed = 4
total_epoch = 100
batch_size = 32
batch_per_epoch = 500


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
model = TimeStampNetwork().to(device)

if device == "cuda":
    model = torch.nn.DataParallel(model)

np_random = np.random.RandomState()
np_random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

loss_function = torch.nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=learn_rate)

epoch_loss = []

'''
for density in range(10):
    for timestamp_index in range(len(timestamp_list[density])):
        print(loaded_music_name[timestamp_to_music_index[density][timestamp_index]])

        input_mel = input_mel_list[timestamp_to_music_index[density][timestamp_index]]
        true_timestamp = timestamp_list[density][timestamp_index]
        density_vector = np.array([0] * 10)
        density_vector[density] = 1
        density_matrix = np.tile(density_vector, (input_mel.shape[0], 1))

        input_mel = torch.Tensor(input_mel).to(device)
        density_matrix = torch.Tensor(density_matrix).to(device)
        true_timestamp = torch.Tensor(true_timestamp).to(device)

        

        output = model.forward([input_mel, density_matrix])
        loss = loss_function(output.squeeze(), true_timestamp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

'''
for epoch in range(total_epoch):
    instance_loss = []
    for instance in range(batch_per_epoch):
        density = np_random.randint(0, len(timestamp_list))

        while True:
            if len(timestamp_list[density]) == 0:
                density = (density + 1) % 10
            else:
                break
        timestamp_index = np_random.randint(0, len(timestamp_list[density]))

        # print(loaded_music_name[timestamp_to_music_index[density][timestamp_index]])

        input_mel = input_mel_list[timestamp_to_music_index[density][timestamp_index]]
        true_timestamp = timestamp_list[density][timestamp_index]
        density_vector = np.array([0] * 10)
        density_vector[density] = 1
        density_matrix = np.tile(density_vector, (input_mel.shape[0], 1))

        input_mel = torch.Tensor(input_mel).to(device)
        density_matrix = torch.Tensor(density_matrix).to(device)
        true_timestamp = torch.Tensor(true_timestamp).to(device)

        

        output = model.forward([input_mel, density_matrix])
        loss = loss_function(output.squeeze(), true_timestamp)
        instance_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del input_mel
        del density_matrix
        del true_timestamp
        del loss
        #print(loss)

    epoch_loss.append(sum(instance_loss))
    print("Epoch " + str(epoch) + "Total Loss: " + str(sum(instance_loss)))
    model_save_path = "../Data/model/" + str(seed) + "/" + str(epoch) + ".model"
    torch.save(model.state_dict(), model_save_path)


np.savetxt("../Data/model/" + str(seed) + "/loss.txt", np.array(epoch_loss))
'''
    for batch in range(batch_per_epoch):
        batch_input = []
        batch_density = np.array([0] * batch_size * 10).reshape(batch_size, 10)
        print(batch_density.shape)
        batch_true = []
        # sample batch data
        for i in range(batch_size):
            density = np_random.randint(0, len(timestamp_list))

            while True:
                if len(timestamp_list[density]) == 0:
                    density = (density + 1) % 10
                else:
                    break

            batch_density[i][density] = 1
            timestamp_index = np_random.randint(0, len(timestamp_list[density]))
            batch_input.append(music_list[timestamp_to_music_index[density][timestamp_index]])
            batch_true.append(timestamp_list[density][timestamp_index])

        # pad the data
        max_length = 0
        for i in batch_true:
            if i.shape[0] > max_length:
                max_length = i.shape[0]

        print(max_length)

        #print(batch_input[0][0].shape)
        #print(batch_input[1][0].shape)
        #print(batch_input[2][0].shape)

        for i in range(batch_size):
            current_length = len(batch_true[i])
            batch_true[i] = np.append(batch_true[i], [-1] * (max_length - current_length))
            fill_up_array = np.array([0] * ((max_length - current_length) * 3 * 80)).reshape(3, max_length - current_length, 80)
            batch_input[i] = np.concatenate((batch_input[i], fill_up_array), axis = 1)

        batch_input = np.array(batch_input)
        batch_true = np.array(batch_true)

        # print(batch_input.shape)
        # print(batch_true.shape)

        # print(model([torch.Tensor(batch_input), torch.Tensor(batch_density)]))

        #print(batch_input[0][0][0].shape)
        #print(batch_input[1][0][0].shape)
        #print(batch_input[2][0][0].shape)
'''






'''
data = np.loadtxt("./-Primary-yuiko-in the Garden-------_1024.dat")

print(data[:,15:30].transpose().shape)

a = np.stack([data[:,0:15].transpose(), data[:,15:30].transpose(), data[:,30:45].transpose()])
print(a.shape)
print(model([torch.Tensor([a]), torch.Tensor([[1,0,0,0,0,0,0,0,0,0]])]))
'''