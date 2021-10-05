# TwoEventHands

In this project, we predict 3D hand poses of two hands based on LNES frames computed using events generated either by an event camera or an event simulator.
For this, we use [MANO](https://mano.is.tue.mpg.de/) parameters.

The network is based on [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) for semantic segmentation and [ResNet50](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) for parameter regression.

## Usage

### Requirements
Requirements include but are not limited to:
* NumPy
* Pillow
* PyTorch
* PyTorch3D
* OpenCV

### Training
To train on all test sequences in the data directory:

`python train.py -m <MODEL>`

### Prediction
To predict a single image:

`python predict.py -m <MODEL> -i <SEQUENCE> -g <GROUND TRUTH>`

Ground truth is optional.

## Notes on parameters and hardware

The scale is typically set to 1 and the batch size to 64.

We use a Geforce GTX 1080 to train the model.
