import torch
import torch.onnx
import onnx
from model.lenet import LeNet

ckpt_path = 'saved_model.pth'
onnx_model = 'saved_model.onnx'

def export(ckpt_path, onnx_model):
  # model object
  model = LeNet()

  # load weights
  ckpt = torch.load(ckpt_path)
  model.load_state_dict(ckpt['state_dict'])

  # create the imput placeholder for the model
  # note: we have to specify the size of a batch of input images
  input_placeholder = torch.randn(1, 1, 28, 28)

  # export
  torch.onnx.export(model, input_placeholder, onnx_model)
  print('{} exported!'.format(onnx_model))

def print_onnx(onnx_model):
  model = onnx.load(onnx_model)
  onnx.checker.check_model(model)
  print('Contents of this model {}:'.format(onnx_model))
  print(onnx.helper.printable_graph(model.graph))

if __name__ == '__main__':
  export(ckpt_path, onnx_model)
  print_onnx(onnx_model)

