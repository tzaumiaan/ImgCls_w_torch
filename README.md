# Image classification with PyTorch

This practical exercise is for understanding how to use PyTorch to train, to save, and to load a neural network model.

## Data preparation
Download the Stanford dogs dataset
```
source download_stanford_dogs_dataset.sh
```
Process the dataset for training
```
python process_stanford_dogs_dataset.py
```

## Training
Run training on training set
```
python train.py
```

## Testing
Run testing on validation set
```
python test.py
```

## Export to ONNX and OpenVINO


## Packages
```
Package       Version
------------- -------
numpy         1.16.4
Pillow        6.0.0
scipy         1.3.0
torch         1.1.0
torchvision   0.3.0
```
