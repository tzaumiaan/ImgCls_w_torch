import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, freeze_feature):
  if freeze_feature:
    for param in model.parameters():
      param.requires_grad = False

def create_model(model_name, n_classes=10, freeze_feature=True):
  input_size = 224
  if model_name == 'resnet18':
    model = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model, freeze_feature)
    n_feat_in = model.fc.in_features
    model.fc = nn.Linear(n_feat_in, n_classes)
  elif model_name == 'vgg19':
    model = models.vgg19(pretrained=True)
    set_parameter_requires_grad(model, freeze_feature)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 1024),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(1024, n_classes)
    )
  elif model_name == 'vgg19_bn':
    model = models.vgg19_bn(pretrained=True)
    set_parameter_requires_grad(model, freeze_feature)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 1024),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(1024, n_classes)
    )
  elif model_name == 'squeezenet1_1':
    model = models.squeezenet1_1(pretrained=True)
    set_parameter_requires_grad(model, freeze_feature)
    model.num_classes = n_classes
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(512, n_classes, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(13, stride=1)
    )
  elif model_name == 'shufflenet_v2_x1_0':
    model = models.shufflenet_v2_x1_0(pretrained=True)
    set_parameter_requires_grad(model, freeze_feature)
    n_feat_in = model.fc.in_features
    model.fc = nn.Linear(n_feat_in, n_classes)
  elif model_name == 'mobilenet_v2':  
    model = models.mobilenet_v2(pretrained=True)
    set_parameter_requires_grad(model, freeze_feature)
    n_feat_in = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(n_feat_in, n_classes)
    )
  elif model_name == 'inception_v3':
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
  model, input_size = create_model('inception_v3', 10)
  print(model)
  model, input_size = create_model('shufflenet_v2_x1_0', 10)
  print(model)
  
