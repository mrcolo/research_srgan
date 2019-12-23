#!/usr/bin/env python

import os
import sys

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

from models import Generator, Discriminator, FeatureExtractor
from config import Config

def check_cuda():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

def maybe_make_dir(directory_name):
    try:
        os.makedirs(directory_name)
    except OSError:
        pass

if __name__ == "__main__":
    # Initial variables
    opt = Config()
    writer = SummaryWriter()
    device = check_cuda()
    maybe_make_dir(opt.out)

    print("Using {}".format(device))
    
    # Transformations
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

    # Fetch Data
    dataset = datasets.ImageFolder(root="./data", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))
    

    #FIXME devicedataloader -> do we need it?
    
    generator = Generator(opt.resBlocks, opt.upSampling)
    discriminator = Discriminator()
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))

    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()
    
    generator = nn.DataParallel(generator)
    generator.to(device)

    discriminator = nn.DataParallel(discriminator)
    discriminator.to(device)
    
    #feature_extractor = nn.DataParallel(feature_extractor)
    feature_extractor.to(device)

    #content_criterion = nn.DataParallel(content_criterion)
    content_criterion.to(device)

    #adversarial_criterion = nn.DataParallel(adversarial_criterion)
    adversarial_criterion.to(device)
   
    ones_const = Variable(torch.ones(opt.batchSize, 1))
    #ones_const = nn.DataParallel(ones_const)
    ones_const.to(device)
   # FIXME you are not doing this
        # content_criterion.cuda()
        # adversarial_criterion.cuda()
        # ones_const = ones_const.cuda()

    optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)

    low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

    # Pre-train generator using raw MSE loss
    print('--> Generator pre-training')
    tensorboard_image_counter = 0

    for epoch in range(opt.generatorPretrainEpochs):
        mean_generator_content_loss = 0.0

        for i, data in enumerate(dataloader):
            # Generate data
            high_res_real, _ = data

            # Downsample images to low resolution
            for j in range(opt.batchSize):
                low_res[j] = scale(high_res_real[j].cpu())
                high_res_real[j] = normalize(high_res_real[j].cpu())

            high_res_real = Variable(high_res_real).to(device)
            high_res_fake = generator(Variable(low_res))

            ######### Train generator #########
            generator.zero_grad()

            generator_content_loss = content_criterion(high_res_fake, high_res_real)
            mean_generator_content_loss += generator_content_loss.sum().item()

            generator_content_loss.sum().backward()
            optim_generator.step()

            ######### Status and display #########
            sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (epoch, opt.generatorPretrainEpochs, i, len(dataloader), generator_content_loss.sum().item()))
        sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f\n' % (epoch, opt.generatorPretrainEpochs, i, len(dataloader), mean_generator_content_loss/len(dataloader)))
        writer.add_scalar('generator_mse_loss', mean_generator_content_loss/len(dataloader), epoch)

    # Do checkpointing
    torch.save(generator.state_dict(), '%s/generator_pretrain.pth' % opt.out)

    # SRGAN training
    optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR*0.1)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR*0.1)

    print('--> SRGAN training')
    for epoch in range(opt.nEpochs):
        mean_generator_content_loss = 0.0
        mean_generator_adversarial_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0

        for i, data in enumerate(dataloader):
            # Generate data
            high_res_real, _ = data

            # Downsample images to low resolution
            for j in range(opt.batchSize):
                low_res[j] = scale(high_res_real[j].cpu())
                high_res_real[j] = normalize(high_res_real[j].cpu())

            # Generate real and fake inputs
            high_res_real = Variable(high_res_real.to(device))
            high_res_fake = generator(Variable(low_res).to(device))
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).to(device)
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).to(device)
        
            ######### Train discriminator #########
            discriminator.zero_grad()

            discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                                adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
            mean_discriminator_loss += discriminator_loss.sum().item()
            
            discriminator_loss.sum().backward()
            optim_discriminator.step()

            ######### Train generator #########
            generator.zero_grad()

            real_features = Variable(feature_extractor(high_res_real).data)
            fake_features = feature_extractor(high_res_fake)

            generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
            mean_generator_content_loss += generator_content_loss.sum().item()
            generator_adversarial_loss = adversarial_criterion(discriminator(Variable(high_res_fake)), ones_const.to(device))
            mean_generator_adversarial_loss += generator_adversarial_loss.sum().item()

            generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
            mean_generator_total_loss += generator_total_loss.sum().item()
            
            generator_total_loss.sum().backward()
            optim_generator.step()   
            
            ######### Status and display #########
            sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (epoch, opt.nEpochs, i, len(dataloader),
            discriminator_loss.item(), generator_content_loss.sum().item(), generator_adversarial_loss.sum().item(), generator_total_loss.sum().item()))

        sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (epoch, opt.nEpochs, i, len(dataloader),
        mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
        mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))
        
        writer.add_scalar('generator_total_loss', mean_generator_total_loss/len(dataloader), epoch)
        writer.add_scalar('generator_adversarial_loss', mean_generator_adversarial_loss/len(dataloader), epoch)
        writer.add_scalar('generator_content_loss', mean_generator_content_loss/len(dataloader), epoch)
        writer.add_scalar('discriminator_loss', mean_discriminator_loss/len(dataloader), epoch)

        # Do checkpointing
        torch.save(generator.state_dict(), '%s/generator_final.pth' % opt.out)
        torch.save(discriminator.state_dict(), '%s/discriminator_final.pth' % opt.out)
