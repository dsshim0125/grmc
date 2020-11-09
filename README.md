## Learning a Geometric Representation for Data-Efficient Depth Estimation via Gradient Field and Contrastive Loss

### IEEE ICRA 2021 Submission 
This repo provides an official PyTorch implementation of the paper submitted to IEEE ICRA 2021, [https://arxiv.org/abs/2011.03207](https://arxiv.org/abs/2011.03207) and pretrained models.


### Results
<p align="center">
	<img src="figs/result.png" alt="photo not available" width="75%" height="75%">
  <img src="figs/table.png" alt="photo not available" width="70%" height="70%">
</p>

### Setup
It is recommended to create a new Anaconda virtual environment for reproduction or evaluation with pretrained models.


```bash
conda create -n grmc python==3.5.6
conda activate grmc
conda install pytorch=1.4.0 torchvision=0.5.0 -c pytorch
```

We ran our experiments with PyTorch 1.4.0, CUDA 10.2, Python 3.5.6 and Ubuntu 18.04. Usage of higher or lower version of PyTorch seems to show incompatible to our pre-trained model.

```bash
pip install pillow==5.2.0 opencv-contrib-python
```
If you do not use Anaconda environment, please use pip3 rather than pip for dependencies with Python3.

### Dataset

Download the subset of preprocessed NYU Depth v2 (50k) [here](https://drive.google.com/drive/folders/1TzwfNA5JRFTPO-kHMU___kILmOEodoBo) for both training and inference from DenseDepth [github](https://github.com/ialhashim/DenseDepth).

### Pretraining Encoder

Any parametric model can be trained with our proposed self-supervised algorithm, and we provide pretrained models of [ResNet-50](https://drive.google.com/file/d/1v-_Egjs_0d8paTo-GpXQPgceoivEw3qd/view?usp=sharing) and [DenseNet-161](https://drive.google.com/file/d/145XimL8EM6D2fumx70CZvOq1tGpv4svY/view?usp=sharing) as the encoder of depth estimation network.


| Encoder  |  batch_size  |
|----------|:--:|
|DenseNet-161| 16 |
|ResNet-50| 64|

```bash
python encoder_pretrain.py --encoder_type --layers --b
```

### Training

| Model  |  Encoder | batch_size|
|----------|:------:|:--:|
|[DenseDepth](https://arxiv.org/abs/1812.11941)| DenseNet-161 |8|
|[FCRN](https://arxiv.org/abs/1606.00373)| ResNet-50|8|
```bash
python train.py --encoder_type --layers --bs
```


### Evaluation

```bash
python evaluate_pytorch.py --model_type --layers
```
