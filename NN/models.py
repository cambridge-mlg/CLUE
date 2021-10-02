from __future__ import division
import torch.nn as nn
import torch.nn.functional as F


class MLP_gauss(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP_gauss, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 2*output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        x = self.block(x)
        mu = x[:, :self.output_dim]
        sigma = F.softplus(x[:, self.output_dim:])
        return mu, sigma


class MLP(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        return self.block(x)


class MNIST_small_cnn(nn.Module):
    def __init__(self,):
        super(MNIST_small_cnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 600)
        self.fc2 = nn.Linear(600, self.output_dim)

        # choose your non linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        #         x = x.view(-1, self.input_dim) # view(batch_size, input_dim)
        x = self.conv1(x)
        x = self.act(x)
        # -----------------
        x = self.conv2(x)
        x = self.act(x)
        # -----------------
        x = x.view(-1, 7 * 7 * 64)
        # -----------------
        x = self.fc1(x)
        x = self.act(x)
        # -----------------
        y = self.fc2(x)

        return y
