import torch
import torch.nn as nn

class TimeStampNetwork(nn.Module):
    def __init__(self):
        super(TimeStampNetwork, self).__init__()

        self.model_cnn = nn.Sequential()
        self.model_blstm = nn.Sequential()
        self.model_tanh = nn.Sequential()
        self.model_fc = nn.Sequential()

        self.model_cnn.add_module("Conv2d_1", nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3))
        self.model_cnn.add_module("Relu_1", nn.ReLU())
        self.model_cnn.add_module("Pooling_1", nn.MaxPool2d(kernel_size=(1,3)))
        self.model_cnn.add_module("Conv2d_2", nn.Conv2d(in_channels=10,out_channels=20,kernel_size=3))
        self.model_cnn.add_module("Relu_2", nn.ReLU())
        self.model_cnn.add_module("Pooling_2", nn.MaxPool2d(kernel_size=(1,3)))

        self.model_blstm.add_module("Blstm_1", nn.LSTM(input_size=11 * 8 * 20 + 10, hidden_size=200, bidirectional=True))

        self.model_tanh.add_module("tanh_1", nn.Tanh())

        self.model_fc.add_module("Fc_1", nn.Linear(in_features=400,out_features=256))
        self.model_fc.add_module("Relu_3", nn.ReLU())
        self.model_fc.add_module("Fc_2", nn.Linear(in_features=256,out_features=128))
        self.model_fc.add_module("Relu_4", nn.ReLU())
        self.model_fc.add_module("Fc_3", nn.Linear(in_features=128,out_features=1))
        self.model_fc.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, input):
        lstm_in = self.model_cnn(input[0])
        lstm_in = torch.cat((lstm_in.view(len(input[0]), 1, -1), input[1].view(len(input[0]), 1 ,10)), dim=2)
        lstm_out, _ = self.model_blstm(lstm_in)
        lstm_out = self.model_tanh(lstm_out)
        output = self.model_fc(lstm_out)
        return output