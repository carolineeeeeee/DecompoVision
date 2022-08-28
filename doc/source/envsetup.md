# Environment Setup for DecompoVision

## Setup
Recommended python version: Python 3.10.4

1. First clone the github repository from here: TODO link

2. Download PASCAL VOC data set from here: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit. Under **Development Kit** click on *training/validation data*.
   
The downloaded PASCAL VOC dataset, once unzip, should have the following structure
```
> tree -L 3 datasets/VOCdevkit
VOCdevkit
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    └── SegmentationObject

```
Then, update the path to the directory `VOCdevkit` of the PASCAL VOC dataset as `VOC_ROOT` in `checking/src/constant.py`.

2. Download model weights from here: https://drive.google.com/drive/u/2/folders/1X5adPQ0d5V38rgcjGXUy4QrrVOUZHpkR and put them in a folder called `model_weights` under `checking`.

## Dependencies: detectron2
[detectron2 by facebookresearch](https://github.com/facebookresearch/detectron2) is imported as a submodule under `checking`.

There are some things you need to do before using it.

1. Since there is no `requirements.txt` for detectron2, we list some dependencies here.

   ````bash
   pip install numpy
   pip install opencv-python
   pip install tqdm
   pip install fvcore
   pip install cloudpickle
   pip install omegaconf
   # install pytorch based on your OS and hardware
   **```**

   ````

2. Run `git submodule update --init` to download the detectron2 source code
3. Run `python setup.py build develop`
4. If you need to write any code that depends on this in this folder, make sure to add the following code at the top of your python script
   ```python
   import os
   import sys
   root = os.path.dirname(os.path.abspath(__file__))
   sys.path.insert(0, root)
   ```

## Estimate Human Tolerated Thresholds

File needed: `checking/estimate_thresholds.py`  
To select the transformation and object class to be estimated, update lines 14 and 15:
```python
CLS = 'person'
TRANSFORMATION = 'frost'
```
This file will read existing experiment results stored in `experiment/experiment_results`

**Example**: In the directory `checking`, run
`python estimate_thresholds.py` will print thresholds estimated for the object class CLS and transformation TRANSFORMATION.


## Checking Requirement Satisfaction

### 1. Bootstrapping

The file `checking/run_bootstrap_pascal_voc.py` is created for generating transformed bootstrap dataset for Pascal VOC dataset. 

This file takes 2 command line arguments:
1. -t or --transformation, specifying the transformation to use

2. -th or --threshold, specifying the human tolerated threshold reused to bound the amount of transformation applied. 

To sepcify the number of batches and number of images per batch, see line 23:

```python
bootstrap_df = bootstrap.bootstrap(image_info_df, 50, 200, args.transformation, float(args.threshold), bootstrap_path)
```
The first number is number of batches, second one is number of images per batch. The default is 50 batches of 200 images, update if needed.

**Example**: In the directory `checking`, run
`python run_bootstrap_pascal_voc.py -t frost -th 0.9` to bootstrap images with frost under the threshold 0.9.


### 2. Run Evaluation for a Transformation and a Threshold

File: `checking/eval_pascal_voc.py`

This step obtains MVC outputs on the bootstrapped images and calculates the metrics. **Note** that this must be run after bootstrapping.

The file takes in 4 command line arguments:
1. -t or --transformation, specifying the transformation to check. 
2. -v or --vision_task, specifying the complex vision task to use, select from ['D', 'I'], 'D' is for object detection and 'I' is for instance segmentation.
3. -th or --threshold, the threshold used during bootstrap, this is used to locate the specific bootstrap images generated with this threshold.
4. -r or --read_only. We provide the MVC output generated for our evaluation RQ2, setting this one to 1 will skip running MVC prediction and read existing MVC results form these files. 
Download from here and put them in the directory `results_csv`: https://drive.google.com/drive/u/2/folders/1PtaaLHT9dKkS2nbNCT77gfCVXD3IwmTP

This file will save results in pickle files. 

**Example**: In the directory `checking`, run
`python eval_pascal_voc.py -t frost -th 0.9 -v D -r 1` will compute the evaluation metrics for object detection using images bootstrapped with frost under threshold 0.9. Since `-r 1` is specified, it will read existing MVC output file from `results_csv/frost_0.9` rather than running MVC prediction.

### 3. Run All Thresholds for Transformations frost and brightness
File: `run_all.sh` and `checking/print_results.py`

`run_bootstrap_pascal_voc.py` runs evaluation per transformation and threshold. `run_all.sh` runs all thresholds for the two transformations used in the paper: frost and brightness.

First make `run_all.sh` executable by running `chmod 777 run_all.sh`, then run `./run_all.sh`. 

After `run_all.sh` finishes, use `print_results.py` to print the reliability distances generated. 

