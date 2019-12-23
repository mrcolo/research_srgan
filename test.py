#!/usr/bin/env python

import argparse
import sys
import os
from math import log10

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import Generator, Discriminator, FeatureExtractor
from config import Config

import pytorch_ssim

def check_cuda():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

if __name__ == '__main__':
    opt = Config()
    device = check_cuda()

    try:
        os.makedirs('output')
    except OSError:
        pass

    transform = transforms.Compose([transforms.RandomCrop(opt.imageSize*opt.upSampling),
                                    transforms.ToTensor()])

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])

    scale = transforms.Compose([transforms.ToPILImage(),
                                transforms.Scale(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                    std = [0.229, 0.224, 0.225])
                                ])

    # Equivalent to un-normalizing ImageNet (for correct visualization)
    unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

    dataset = datasets.ImageFolder(root="./test", transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                             shuffle=False, num_workers=int(opt.workers))

    generator = Generator(opt.resBlocks, opt.upSampling)
    generator = nn.DataParallel(generator).to(device)
    generator.load_state_dict(torch.load("./checkpoints/generator_final.pth"))

    content_criterion = nn.MSELoss()
    content_criterion.to(device)
    
    target_real = Variable(torch.ones(4,1)).to(device)
    target_fake = Variable(torch.zeros(4,1)).to(device)

    low_res = torch.FloatTensor(4, 3, opt.imageSize, opt.imageSize)

    content_criterion = nn.MSELoss()

    # Set evaluation mode (not training)
    generator.eval()

    for i, data in enumerate(dataloader):
        # Generate data
        high_res_real, _ = data

        # Downsample images to low resolution
        for j in range(4):
            low_res[j] = scale(high_res_real[j].cpu())
            high_res_real[j] = normalize(high_res_real[j].cpu())

        high_res_real = Variable(high_res_real).to(device)
        high_res_fake = generator(Variable(low_res))

        for j in range(4):
            low_res[j] = unnormalize(low_res[j].cpu())
            high_res_real[j] = unnormalize(high_res_real[j].cpu())
            high_res_fake[j] = unnormalize(high_res_fake[j].cpu())
        
        mse = content_criterion(high_res_fake, high_res_real)
        psnr = 10 * log10(1 / mse.item())
        ssim = pytorch_ssim.ssim(high_res_fake, high_res_real)
        print("[{}] --> PSNR ({})\n    --> SSIM ({}%)".format(i, psnr, int(ssim*100)))

        save_image(torchvision.utils.make_grid(low_res), 'output/' + str(i) + '_lowres.png')
        save_image(torchvision.utils.make_grid(high_res_real), 'output/' + str(i) + '_highres.png')
        save_image(torchvision.utils.make_grid(high_res_fake.data), 'output/' + str(i) + '_superres.png')

