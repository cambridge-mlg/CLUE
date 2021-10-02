from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
from src.layers import *

class small_MNIST_Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class small_MNIST_unFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 128, 4, 4)


# ResNet MNIST conv model with class --> These are a bit strange because of combining categorical with images

class MNIST_recognition_resnet(nn.Module):
    def __init__(self, latent_dim, targets=False):
        super(MNIST_recognition_resnet, self).__init__()

        width_mul = 3

        self.targets = targets
        if targets:
            self.extra_width = 10
        else:
            self.extra_width = 0

        self.resnet = nn.Sequential(nn.Conv2d(1, 32, kernel_size=1, padding=0, stride=1),  # 28x28 --32
                       ResBlock(outer_dim=32, inner_dim=8*width_mul), ResBlock(outer_dim=32, inner_dim=8*width_mul),
                       nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),  # 14x14 --64
                       ResBlock(outer_dim=64, inner_dim=16*width_mul), ResBlock(outer_dim=64, inner_dim=16*width_mul),
                       nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),  # 7x7 --128
                       ResBlock(outer_dim=128, inner_dim=32*width_mul), ResBlock(outer_dim=128, inner_dim=32*width_mul),
                       nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),  # 4x4 --128
                       small_MNIST_Flatten(), nn.LeakyReLU(inplace=True), nn.BatchNorm1d(num_features=4*4*128))

        self.MLP = nn.Sequential(nn.Linear(4*4*128+self.extra_width, 128),
                            preact_leaky_MLPBlock(128), preact_leaky_MLPBlock(128),
                            nn.LeakyReLU(), nn.BatchNorm1d(num_features=128), nn.Linear(128, latent_dim*2))

    def forward(self, x):

        if self.targets:
            x, y = x[:, :-10], x[:, -10:]
        x = x.view(x.shape[0], 1, 28, 28)

        x = self.resnet(x)

        if self.targets:
            x = torch.cat([x, y], dim=1)
        return self.MLP(x)


class MNIST_prior_resnet(nn.Module):

    def __init__(self, latent_dim, targets=False):
        super(MNIST_prior_resnet, self).__init__()

        width_mul = 3

        self.targets = targets
        if targets:
            self.extra_width = 10 * 2 # double is there for the mask
        else:
            self.extra_width = 0

        self.resnet = nn.Sequential(nn.Conv2d(2*1, 32, kernel_size=1, padding=0, stride=1),  # 28x28 --32
                                    ResBlock(outer_dim=32, inner_dim=8 * width_mul),
                                    ResBlock(outer_dim=32, inner_dim=8 * width_mul),
                                    nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),  # 14x14 --64
                                    ResBlock(outer_dim=64, inner_dim=16 * width_mul),
                                    ResBlock(outer_dim=64, inner_dim=16 * width_mul),
                                    nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),  # 7x7 --128
                                    ResBlock(outer_dim=128, inner_dim=32 * width_mul),
                                    ResBlock(outer_dim=128, inner_dim=32 * width_mul),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),  # 4x4 --128
                                    small_MNIST_Flatten(), nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm1d(num_features=4 * 4 * 128))

        self.MLP = nn.Sequential(nn.Linear(4 * 4 * 128 + self.extra_width, 128),
                                 preact_leaky_MLPBlock(128), preact_leaky_MLPBlock(128),
                                 nn.LeakyReLU(), nn.BatchNorm1d(num_features=128), nn.Linear(128, latent_dim * 2))

    def forward(self, x):

        mask = x[:, 784+int(self.extra_width/2):]
        x = x[:, :784+int(self.extra_width/2)]

        if self.targets:
            mask, mask_y = mask[:, :-int(self.extra_width/2)], mask[:, -int(self.extra_width/2):]
            x, y = x[:, :-int(self.extra_width/2)], x[:, -int(self.extra_width/2):]

        mask = mask.view(x.shape[0], 1, 28, 28)
        x = x.view(x.shape[0], 1, 28, 28)

        x = torch.cat([x, mask], dim=1)
        x = self.resnet(x)

        if self.targets:
            x = torch.cat([x, y, mask_y], dim=1)
        return self.MLP(x)


class MNIST_generator_resnet(nn.Module):
    def __init__(self, latent_dim, targets=False):
        super(MNIST_generator_resnet, self).__init__()

        width_mul = 3

        self.targets = targets
        if targets:
            self.extra_width = 10
        else:
            self.extra_width = 0

        self.MLP = nn.Sequential(nn.Linear(latent_dim, 128), nn.BatchNorm1d(num_features=128), nn.LeakyReLU(),
                                 leaky_MLPBlock(128), leaky_MLPBlock(128),
                                 nn.Linear(128, 4 * 4 * 128 + self.extra_width),)

        self.resnet = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm1d(num_features=4*4*128), small_MNIST_unFlatten(),  # 4x4 --128
                         nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2),  # 7x7 --128
                         ResBlock(outer_dim=128, inner_dim=32*width_mul), ResBlock(outer_dim=128, inner_dim=32*width_mul),
                         nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2),  # 14x14 --64
                         ResBlock(outer_dim=64, inner_dim=16*width_mul), ResBlock(outer_dim=64, inner_dim=16*width_mul),
                         nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2),  # 28x28 --32
                         ResBlock(outer_dim=32, inner_dim=8*width_mul), ResBlock(outer_dim=32, inner_dim=8*width_mul),
                         nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1),)  # 28x28 --1)

    def forward(self, x):

        x = self.MLP(x)
        if self.targets:
            x, y = x[:, :-10], x[:, -10:]

        x = self.resnet(x)

        x = x.view(x.shape[0], -1)
        if self.targets:
            x = torch.cat([x, y], dim=1)
        return x


# Resnet model for doodle

class doodle_recognition_resnet(nn.Module):
    def __init__(self, latent_dim, targets=False, Nclass=0):
        super(doodle_recognition_resnet, self).__init__()

        width_mul = 3

        self.targets = targets
        if targets:
            self.extra_width = Nclass
        else:
            self.extra_width = 0

        self.resnet = nn.Sequential(nn.Conv2d(1, 32, kernel_size=1, padding=0, stride=1),  # 64x64 --32
                       ResBlock(outer_dim=32, inner_dim=8*width_mul), ResBlock(outer_dim=32, inner_dim=8*width_mul),
                       nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),  # 32x32 --64
                       ResBlock(outer_dim=64, inner_dim=16*width_mul), ResBlock(outer_dim=64, inner_dim=16*width_mul),
                       nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),  # 16x16 --128
                       ResBlock(outer_dim=128, inner_dim=32*width_mul), ResBlock(outer_dim=128, inner_dim=32*width_mul),
                       nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2),  # 8x8 --128
                       ResBlock(outer_dim=128, inner_dim=32 * width_mul), ResBlock(outer_dim=128, inner_dim=32 * width_mul),
                       nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2),  # 4x4 --128
                       small_MNIST_Flatten(), nn.LeakyReLU(inplace=True), nn.BatchNorm1d(num_features=4 * 4 * 128))

        self.MLP = nn.Sequential(nn.Linear(4*4*128+self.extra_width, 128),
                            preact_leaky_MLPBlock(128), preact_leaky_MLPBlock(128),
                            nn.LeakyReLU(), nn.BatchNorm1d(num_features=128), nn.Linear(128, latent_dim*2))

    def forward(self, x):

        if self.targets:
            x, y = x[:, :-self.extra_width], x[:, -self.extra_width:]
        x = x.view(x.shape[0], 1, 64, 64)

        x = self.resnet(x)

        if self.targets:
            x = torch.cat([x, y], dim=1)
        return self.MLP(x)


class doodle_prior_resnet(nn.Module):

    def __init__(self, latent_dim, targets=False, Nclass=0):
        super(doodle_prior_resnet, self).__init__()

        width_mul = 3

        self.targets = targets
        if targets:
            self.extra_width = Nclass * 2  # double is there for the mask
        else:
            self.extra_width = 0

        self.resnet = nn.Sequential(nn.Conv2d(2*1, 32, kernel_size=1, padding=0, stride=1),  # 64x64 --32
                                    ResBlock(outer_dim=32, inner_dim=8 * width_mul),
                                    ResBlock(outer_dim=32, inner_dim=8 * width_mul),
                                    nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),  # 32x32 --64
                                    ResBlock(outer_dim=64, inner_dim=16 * width_mul),
                                    ResBlock(outer_dim=64, inner_dim=16 * width_mul),
                                    nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),  # 16x16 --128
                                    ResBlock(outer_dim=128, inner_dim=32 * width_mul),
                                    ResBlock(outer_dim=128, inner_dim=32 * width_mul),
                                    nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2),  # 8x8 --128
                                    ResBlock(outer_dim=128, inner_dim=32 * width_mul),
                                    ResBlock(outer_dim=128, inner_dim=32 * width_mul),
                                    nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2),  # 4x4 --128
                                    small_MNIST_Flatten(), nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm1d(num_features=4 * 4 * 128))

        self.MLP = nn.Sequential(nn.Linear(4 * 4 * 128 + self.extra_width, 128),
                                 preact_leaky_MLPBlock(128), preact_leaky_MLPBlock(128),
                                 nn.LeakyReLU(), nn.BatchNorm1d(num_features=128), nn.Linear(128, latent_dim * 2))

    def forward(self, x):

        mask = x[:, 64*64+int(self.extra_width/2):]
        x = x[:, :64*64+int(self.extra_width/2)]

        if self.targets:
            mask, mask_y = mask[:, :-int(self.extra_width/2)], mask[:, -int(self.extra_width/2):]
            x, y = x[:, :-int(self.extra_width/2)], x[:, -int(self.extra_width/2):]

        mask = mask.view(x.shape[0], 1, 64, 64)
        x = x.view(x.shape[0], 1, 64, 64)

        x = torch.cat([x, mask], dim=1)
        x = self.resnet(x)

        if self.targets:
            x = torch.cat([x, y, mask_y], dim=1)
        return self.MLP(x)


class doodle_generator_resnet(nn.Module):
    def __init__(self, latent_dim, targets=False, Nclass=0):
        super(doodle_generator_resnet, self).__init__()

        width_mul = 3

        self.targets = targets
        if targets:
            self.extra_width = Nclass
        else:
            self.extra_width = 0

        self.MLP = nn.Sequential(nn.Linear(latent_dim, 128), nn.BatchNorm1d(num_features=128), nn.LeakyReLU(),
                                 leaky_MLPBlock(128), leaky_MLPBlock(128),
                                 nn.Linear(128, 4 * 4 * 128 + self.extra_width),)

        self.resnet = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm1d(num_features=4*4*128), small_MNIST_unFlatten(),  # 4x4 --128
                         nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2),  # 4x4 --128
                         ResBlock(outer_dim=128, inner_dim=32 * width_mul), ResBlock(outer_dim=128, inner_dim=32 * width_mul),
                         nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2),  # 8x8 --64
                         ResBlock(outer_dim=128, inner_dim=32*width_mul), ResBlock(outer_dim=128, inner_dim=32*width_mul),
                         nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2),  # 16x16 --64
                         ResBlock(outer_dim=64, inner_dim=16*width_mul), ResBlock(outer_dim=64, inner_dim=16*width_mul),
                         nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2),  # 32x32 --32
                         ResBlock(outer_dim=32, inner_dim=8*width_mul), ResBlock(outer_dim=32, inner_dim=8*width_mul),
                         nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1),)  # 64x64 --1)

    def forward(self, x):

        x = self.MLP(x)
        if self.targets:
            x, y = x[:, :-self.extra_width], x[:, -self.extra_width:]

        x = self.resnet(x)

        x = x.view(x.shape[0], -1)
        if self.targets:
            x = torch.cat([x, y], dim=1)
        return x

# FC Networks


class MLP_prior_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_prior_net, self).__init__()
        # input layer
        proposal_layers = [nn.Linear(input_dim*2, width), nn.ReLU(), nn.BatchNorm1d(num_features=width)]
        # body
        for i in range(depth-1):
            proposal_layers.append(MLPBlock(width))
        # output layer
        proposal_layers.append(
            nn.Linear(width, latent_dim * 2)
        )

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MLP_recognition_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_recognition_net, self).__init__()
        # input layer
        proposal_layers = [nn.Linear(input_dim, width), nn.ReLU(), nn.BatchNorm1d(num_features=width)]
        # body
        for i in range(depth-1):
            proposal_layers.append(MLPBlock(width))
        # output layer
        proposal_layers.append(
            nn.Linear(width, latent_dim * 2)
        )

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MLP_generator_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_generator_net, self).__init__()
        # input layer
        generative_layers = [nn.Linear(latent_dim, width), nn.LeakyReLU(), nn.BatchNorm1d(num_features=width)]
        # body
        for i in range(depth-1):
            generative_layers.append(
                # skip-connection from prior network to generative network
                leaky_MLPBlock(width))
        # output layer
        generative_layers.extend([
            nn.Linear(width,
                      input_dim),
        ])
        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)

## Fully linear residual path preact models


class MLP_preact_prior_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_preact_prior_net, self).__init__()
        # input layer
        proposal_layers = [nn.Linear(input_dim*2, width)]
        # body
        for i in range(depth-1):
            proposal_layers.append(preact_leaky_MLPBlock(width))
        # output layer
        proposal_layers.extend([nn.LeakyReLU(), nn.BatchNorm1d(num_features=width), nn.Linear(width, latent_dim * 2)])


        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MLP_preact_recognition_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_preact_recognition_net, self).__init__()
        # input layer
        proposal_layers = [nn.Linear(input_dim, width)]
        # body
        for i in range(depth-1):
            proposal_layers.append(preact_leaky_MLPBlock(width))
        # output layer
        proposal_layers.extend([nn.LeakyReLU(), nn.BatchNorm1d(num_features=width), nn.Linear(width, latent_dim * 2)])

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MLP_preact_generator_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_preact_generator_net, self).__init__()
        # input layer
        generative_layers = [nn.Linear(latent_dim, width)]
        # body
        for i in range(depth-1):
            generative_layers.append(
                # skip-connection from prior network to generative network
                preact_leaky_MLPBlock(width))
        # output layer
        generative_layers.extend([
            nn.LeakyReLU(), nn.BatchNorm1d(num_features=width), nn.Linear(width, input_dim),
        ])
        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)