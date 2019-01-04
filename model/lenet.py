import torch.nn as nn
import torch.nn.functional as fn

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)
    self.fc1 = nn.Linear(4*4*50, 500)
    self.fc2 = nn.Linear(500, 10)

  def forward(self, x):
    x = fn.relu(self.conv1(x))
    x = fn.max_pool2d(x, 2, 2)
    x = fn.relu(self.conv2(x))
    x = fn.max_pool2d(x, 2, 2)
    x = x.view(-1, 4*4*50)
    x = fn.relu(self.fc1(x))
    x = self.fc2(x)
    return x
  
  def name(self):
    return "LeNet"
