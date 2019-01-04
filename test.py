from data_utils import load_data
from model.lenet import LeNet
from train import apply_cuda

from datetime import datetime
import torch

ckpt_path = 'saved_model.pth'

def test():
  # model definition
  model = apply_cuda(LeNet())

  # load weights
  ckpt = torch.load(ckpt_path)
  model.load_state_dict(ckpt['state_dict'])
  
  # data source
  test_loader = load_data(mode='test', data_folder='data')
  images, labels = iter(test_loader).next()
  images, labels = apply_cuda(images), apply_cuda(labels)
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

