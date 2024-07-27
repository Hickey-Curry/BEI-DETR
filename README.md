# BEI-DETR

# Usage - Object detection
There are no extra compiled components in BEI-DETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/Hickey-Curry/BEI-DETR.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

(optional) to work with panoptic install panopticapi:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Data preparation
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  train-EEG     # train EEG data
  train-EYE     # train EYE images
  val2017/      # val images
  val-EEG       # val EEG data
  val-EYE       # val EYE images
```

## Training
To train BEI-DETR with 8 gpus for 500 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```

## Evaluation
To evaluate BEI-DETR with a single GPU run:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume /path/to/best.pth --coco_path /path/to/coco
```
