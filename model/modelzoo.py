import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, freeze_feature):
  if freeze_feature:
    for param in model.parameters():
      param.requires_grad = False

def create_model(model_name, n_classes, freeze_feature=True):
  input_size = 224
  if model_name == "resnet18":
    model = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model, freeze_feature)
    n_feat_in = model.fc.in_features
    model.fc = nn.Linear(n_feat_in, n_classes)
  elif model_name == "inception_v3":
    # it expects (299,299) sized images and has auxiliary output
    input_size = 299
    model = models.inception_v3(pretrained=True)
    set_parameter_requires_grad(model, freeze_feature)
    # Handle the auxilary net
    n_feat_in = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(n_feat_in, n_classes)
    # Handle the primary net
    n_feat_in = model.fc.in_features
    model.fc = nn.Linear(n_feat_in,n_classes)
  else:
    raise (ValueError, 'invalid model')
  return model, input_size


# unit test
if __name__=='__main__':
  model, input_size = create_model('resnet', 10)
  print(model)
