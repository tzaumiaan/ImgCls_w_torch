import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5, 1)
    self.conv2 = nn.Conv2d(6, 16, 5, 1)
    self.fc1 = nn.Linear(7*7*16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.dropout2 = nn.Dropout(0.5)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.pad(x, (2,2,2,2), mode='replicate')
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.pad(x, (2,2,2,2), mode='replicate')
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 7*7*16)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.dropout2(x)
    x = self.fc3(x)
    return x
  
  def name(self):
    return "LeNet"
