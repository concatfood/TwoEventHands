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

`python train.py -b <BATCH SIZE> -s <SCALE>`

### Prediction
To predict a single image:

`python predict.py -m <MODEL> -s <SCALE> -i <SEQUENCE> -f <FRAME>`

To predict multiple images:

`python predict_all.py -m <MODEL> -s <SCALE> -i <DIRECTORY OF SEQUENCES>`

## Notes on parameters and hardware

The scale is typically set to 1 and the batch size to 16.

We use a Geforce GTX 1080 to train the model.
