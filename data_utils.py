import torch
from torchvision import transforms
import torchvision.datasets as dset
import os

def load_data(mode='train', batch_size=10, data_folder='data', dataset_name='mnist'):
  if dataset_name=='mnist':
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
  elif dataset_name=='sdd':
    target_size = 224
    data_folder = os.path.join(data_folder, dataset_name, mode)
    trans = {
      'train': transforms.Compose([
        transforms.RandomSizedCrop(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
      'test': transforms.Compose([
        # Higher scale-up for inception
        transforms.Scale(int(target_size/224*256)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }
    if mode == 'train':
      dataset = dset.ImageFolder(root=data_folder, transform=trans['train']) 
      loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    elif mode == 'test':
      dataset = dset.ImageFolder(root=data_folder, transform=trans['test']) 
      loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
  else:
    raise ValueError

  return loader

# unit test
if __name__=='__main__':
  loader = load_data(mode='train', batch_size=10, data_folder='data', dataset_name='sdd')
