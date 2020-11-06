import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
import PIL
import cv2
from scipy import io as scipy_io


def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}

def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    val_index = int(len(nyu2_train)*0.8)
    nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_val = nyu2_train[val_index:]
    nyu2_train = nyu2_train[:val_index]

    from sklearn.utils import shuffle
    nyu2_train = shuffle(nyu2_train, random_state=0)

    #if True: nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train, nyu2_val, nyu2_test


def loadZipToMem_test(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    from sklearn.utils import shuffle
    nyu2_test = shuffle(nyu2_test, random_state=0)

    #if True: nyu2_test = nyu2_test[:40]

    print('Loaded ({0}).'.format(len(nyu2_test)))
    return data, nyu2_test

class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = Image.open( BytesIO(self.data[sample[1]]) )
        sample = {'image': image, 'depth': depth}
        if self.transform: sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class ToTensor(object):
    def __init__(self,is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        image = self.to_tensor(image)

        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:            
            depth = self.to_tensor(depth).float() * 1000
        
        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])

def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])

def getTrainingTestingData(batch_size):
    #data, nyu2_train = loadZipToMem('nyu_data.zip')
    data, nyu2_train, nyu2_val, nyu2_test = loadZipToMem('/media/dsshim/nyu_v2/nyu_data.zip')

    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_val = depthDatasetMemory(data, nyu2_val, transform=getNoTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_test, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size, shuffle=False)


class MoCoData(Dataset):

    def __init__(self, transform=None, type='train'):
        self.root_dir = '/media/dsshim/nyu_v2/'
        self.transform = transform

        if type == 'train':
            self.df = pd.DataFrame(pd.read_csv(self.root_dir + 'data/nyu2_train.csv', header=None))

        elif type == 'test':
            self.df = pd.DataFrame(pd.read_csv(self.root_dir + 'data/nyu2_test.csv', header=None))

        self.type = type

    def __len__(self):

        return int(len(self.df))

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])

        img = PIL.Image.open(img_name)

        if self.transform:
            img = self.transform(img)

        img_gray = img.convert('L')
        img = np.array(img)
        img_gray = np.array(img_gray)

        kernels = [np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape(3,3),\
                   np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(3,3)]

        img_v = cv2.filter2D(img_gray, -1, kernels[0])
        img_h = cv2.filter2D(img_gray, -1, kernels[1])

        img = np.array(img)

        edge_mask = cv2.Canny(img, 50, 200)

        magnitude = np.sqrt(img_v**2+img_h**2).astype(np.float64)

        edge = magnitude*edge_mask

        if np.max(edge) ==0:

            img *= 0

        else:

            edge = edge/np.max(edge) * 255


        edge = Image.fromarray(edge)

        img = transforms.ToTensor()(img)
        edge = transforms.ToTensor()(edge)

        edge = edge.expand_as(img)

        return img, edge