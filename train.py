from data_utils import load_data
from model.lenet import LeNet

from datetime import datetime
import torch

ckpt_path = 'saved_model.pth'

def apply_cuda(obj):
  if torch.cuda.is_available():
    obj = obj.cuda()
  return obj

def apply_var(obj):
  return torch.autograd.Variable(obj)

def train(batch_size=50, lr=0.01, data_folder='data', dataset_name='mnist', max_epochs=10, log_freq=100):
  # data source
  train_loader = load_data('train', batch_size, data_folder)
  num_batches = len(train_loader)
  # model definition
  model = apply_cuda(LeNet())
  # optimizer and loss definition
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  
  for epoch in range(max_epochs):
    print(datetime.now(), 'epoch: {}/{}'.format(epoch+1, max_epochs))
    
    # training set
    print('==== training phase ====')
    avg_loss = float(0)
    avg_acc = float(0)
    for step, (images, labels) in enumerate(train_loader):
      optimizer.zero_grad()
      images, labels = apply_cuda(images), apply_cuda(labels)
      images, labels = apply_var(images), apply_var(labels)
      # forward pass
      logits = model(images)
      loss = criterion(logits, labels)
      _, pred = torch.max(logits.data, 1)
      batch_size = labels.data.size()[0]
      match_count = (pred == labels.data).sum()
      accuracy = float(match_count)/float(batch_size)
      avg_loss += loss.item()/float(num_batches)
      avg_acc += accuracy/float(num_batches)
      # backward pass
      loss.backward()
      optimizer.step()
      if (step+1) % log_freq == 0:
        print(
            datetime.now(),
            'training step: {}/{}'.format(step+1, num_batches),
            'loss={:.5f}'.format(loss.item()),
            'acc={:.4f}'.format(accuracy))
    print(
        datetime.now(),
        'training ends with avg loss={:.5f}'.format(avg_loss),
        'and avg acc={:.4f}'.format(avg_acc))
    # validation set
    print('==== validation phase ====')
    # TODO 
    
    # save the model for every epoch
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'avg_loss': avg_loss,
        'avg_acc': avg_acc}, ckpt_path)

if __name__ == '__main__':
  train()

