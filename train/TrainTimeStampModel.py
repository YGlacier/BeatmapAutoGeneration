import sys
sys.path.append("../")
from model.TimeStampNetwork import TimeStampNetwork

import torch
import numpy as np

model = TimeStampNetwork()

data = np.loadtxt("./-Primary-yuiko-in the Garden-------_1024.dat")

print(data[:,15:30].transpose().shape)

a = np.stack([data[:,0:15].transpose(), data[:,15:30].transpose(), data[:,30:45].transpose()])
print(a.shape)
print(model([torch.Tensor([a]), torch.Tensor([[1,0,0,0,0,0,0,0,0,0]])]))