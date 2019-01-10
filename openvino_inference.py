from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np

model_xml = 'saved_model.xml'
model_bin = 'saved_model.bin'
test_dataset = 'data/mnist_test_data.npz'

def load_model(device, model_xml, model_bin):
  plugin = IEPlugin(device=device, plugin_dirs=None)
  net = IENetwork(model=model_xml, weights=model_bin)
  exec_net = plugin.load(network=net)
  return exec_net

def load_input(datafile):
  f = np.load(datafile)
  return f['images'], f['labels']

def inference():
  pass

def main():
  # load model
  exec_net = load_model('CPU', model_xml, model_bin)
  # load input
  images, labels = load_input(test_dataset)



if __name__=='__main__':
  main()

