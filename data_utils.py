import torch
from torchvision import transforms
import torchvision.datasets as dset
import os

def load_data(mode='train', batch_size=10, data_folder='data'):
  if not os.path.exists(data_folder):
    os.mkdir(data_folder)

  trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
  # if not exist, download mnist dataset
  if mode == 'train':
    dataset = dset.MNIST(root=data_folder, train=True, transform=trans, download=True)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
  elif mode == 'test':
    dataset = dset.MNIST(root=data_folder, train=False, transform=trans, download=True)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
  
  return loader
