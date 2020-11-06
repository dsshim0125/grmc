import os
import argparse
import torch
import torch.nn as nn
from model.densedepth import Model
from model.fcrn import ResNet
from loss import ssim, MaskedL1Loss
from data import getTrainingTestingData
from utils import DepthNorm
import tqdm


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Finetuning for depth estimation')
    parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--model_type', default='densedepth', help='type of the depth estimation network')
    parser.add_argument('--layers', default=161, type=int, help='number of layers of encoder')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    args = parser.parse_args()
    # Create model

    if args.model_type == 'densedepth':
        model = Model(layers=args.layers)

    else:
        model = ResNet(layers=args.layers)

    model = model.cuda()
    model = nn.DataParallel(model)

    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )

    batch_size = args.bs

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)


    # Loss
    l1_criterion = nn.L1Loss()
    masked_criterion = MaskedL1Loss()


    for epoch in range(args.epochs):

        model.train()


        for i, sample_batched in enumerate(tqdm.tqdm(train_loader)):
            
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # Normalize depth
            depth_n = DepthNorm( depth )

            # Predict
            output = model(image)

            if args.model_type == 'densedepth':

                l_depth = l1_criterion(output, depth_n)
                l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

                loss = (1.0 * l_ssim) + (0.1 * l_depth)

            else:

                loss = masked_criterion(output, depth_n)

            # Update step
            loss.backward()

            optimizer.step()


        torch.save(model.module.state_dict(), 'checkpoints/%s_%d.pth'%(args.model_type, args.layers))
        print('Epoch:%d Model Saved!'%(epoch+1))



if __name__ == '__main__':
    main()
