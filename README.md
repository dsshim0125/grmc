## Learning a Geometric Representation for Data-Efficient Depth Estimation via Gradient Field and Contrastive Loss

### IEEE ICRA 2021 Submission 
Official PyTorch implementation of the paper.

![Figure](figs/overview.png)

### Introduction
Estimating a depth map from a single RGB image has been investigated widely for localization, mapping, and 3- dimensional object detection. Recent studies on a single-view depth estimation are mostly based on deep Convolutional neural Networks (ConvNets) which require a large amount of training data paired with densely annotated labels. Depth annotation tasks are both expensive and inefficient, so it is inevitable to leverage RGB images which can be collected very easily to boost the performance of ConvNets without depth labels. However, most self-supervised learning algorithms are focused on capturing the semantic information of images to improve the performance in classification or object detection, not in depth estimation. In this paper, we show that existing self-supervised methods do not perform well on depth estimation and propose a gradient-based self-supervised learning algorithm with momentum contrastive loss to help ConvNets extract the geometric information with unlabeled images. As a result, the network can estimate the depth map accurately with a relatively small amount of annotated data. To show that our method is independent of the model structure, we evaluate our method with two different monocular depth estimation algorithms. Our method outperforms the previous state-of-the- art self-supervised learning algorithms and shows the efficiency of labeled data in triple compared to random initialization on the NYU Depth v2 dataset.


### Setup
It is recommended to use Anaconda virtual environment  for reproduction or evaluation with pretrained models.


```bash
conda install pytorch=1.4.0 torchvision=0.5.0 -c pytorch
```

We ran our experiments with PyTorch 1.4.0, CUDA 10.2, Python 3.5.6 and Ubuntu 18.04. Usage of higher or lower version of PyTorch seems to show incompatible to our pre-trained model.

```bash
pip install pillow==5.2.0 opencv-contrib-python
```
If you do not use Anaconda environment, use pip3 rather than pip for dependencies with Python3.
