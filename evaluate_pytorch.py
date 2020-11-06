import os
import glob
import time
import argparse
import torch
from torch import nn
from model.densedepth import Model
from model.fcrn import ResNet
from utils import evaluate
from matplotlib import pyplot as plt


# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model_type', default='densedepth', type=str, help='Depth estimation network for evaluation')
parser.add_argument('--layers', default=161, type=int, help='number of layers of encoder')

args = parser.parse_args()

# Custom object needed for inference and training

# Load test data

print('Loading test data...', end='')
import numpy as np
from zipfile import ZipFile
def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

data = extract_zip('/media/dsshim/nyu_v2/nyu_test.zip')
from io import BytesIO
rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
depth = np.load(BytesIO(data['eigen_test_depth.npy']))
crop = np.load(BytesIO(data['eigen_test_crop.npy']))
print('Test data loaded.\n')




if args.model_type == 'densedepth':
    model = Model()

else:
    model = ResNet(layers=args.layers)


model.load_state_dict(torch.load('checkpoints/%s_%d.pth'%(args.model_type, args.layers)))
model = model.cuda()

model.eval()




start = time.time()
print('Testing...')

e = evaluate(model, rgb, depth, crop, batch_size=1)


end = time.time()
print('\nTest time', end-start, 's')
