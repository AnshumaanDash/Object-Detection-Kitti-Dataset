# Object-Detection-Kitti-Dataset

Kitti Dataset:
    
    @inproceedings{Geiger2012CVPR,
      author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
      title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
      booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2012}
    }

This Datasets contains the Kitti Object Detection Benchmark, created by *Andreas Geiger, Philip Lenz* and *Raquel Urtasun* in the *Proceedings of 2012 CVPR ," Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite"*. The data used contains the object detection part of their different Datasets published for Autonomous Driving. It contains a set of images with their bounding box labels. For more information visit the Website they published the data on (http://www.cvlibs.net/datasets/kitti/).



This code was trained and tested on an Ubuntu 18.04 system with Tesla T4 GPU, using PyTorch 1.3.1 and Torchvision 0.4.2.

## Steps to Train:

1. Run the `script.sh` file - loads the torchvision dependencies
2. Modify the `config.py` file as per your local directory paths
3. Run `main.py` with required arguments

## Steps to Test:

1. Run the `script.sh` file - loads the torchvision dependencies
2. Call the `object_detection_api` from the `prediction.py` file with the test image

## Results:

**mAP (0.50) = 0.944**

![image](https://user-images.githubusercontent.com/26281528/72545168-9ac09a00-38ae-11ea-9104-33cfef4ad65c.png)

## Sample output:

![download](https://user-images.githubusercontent.com/26281528/72544799-ffc7c000-38ad-11ea-809d-c477e083872a.png)


Many thanks to [Francisco Massa](https://github.com/fmassa) for their starter code in the torchvision repo.
