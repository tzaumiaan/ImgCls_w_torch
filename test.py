from data_utils import load_data
from model.lenet import LeNet
from train import apply_cuda

from datetime import datetime
import torch

ckpt_path = 'saved_model.pth'

def export_test_data_to_numpy(images, labels):
  import numpy as np
  np.savez('data/mnist_test_data.npz', images=images, labels=labels)

def test():
  # model definition
  model = apply_cuda(LeNet())

  # load weights
  ckpt = torch.load(ckpt_path)
  model.load_state_dict(ckpt['state_dict'])
  
  # data source
  test_loader = load_data(mode='test', data_folder='data')
  images, labels = iter(test_loader).next()
  export_test_data_to_numpy(images.data.numpy(), labels.data.numpy())
  images, labels = apply_cuda(images), apply_cuda(labels)
  print(datetime.now(), 'test batch with shape {}'.format(images.shape))
  logits = model(images)
  _, pred = torch.max(logits.data, 1)
  data_size = labels.data.size()[0]
  match_count = (pred == labels.data).sum()
  accuracy = float(match_count)/float(data_size)
  print(
      datetime.now(),
      'testing results: acc={:.4f}'.format(accuracy))
  

if __name__ == '__main__':
  test()

