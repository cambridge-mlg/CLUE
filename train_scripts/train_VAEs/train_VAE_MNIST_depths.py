from __future__ import division, print_function 
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % int(sys.argv[1])

N_up = 2
nb_dir = '/'.join(os.getcwd().split('/')[:-N_up])
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


import torch
from torchvision import datasets, transforms
from VAE.MNISTconv_bern import MNISTconv_VAE_bern_net
from VAE.train import train_VAE
from VAE.models import MNIST_generator_resnet, MNIST_recognition_resnet

save_dir = '../saves'

transform_train = transforms.Compose([
    transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
valset = datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)

for latent_dim in [12, 16, 30, 40, 80]: #2, 4, 8,

    batch_size = 256
    nb_epochs = 300
    lr = 7e-4
    early_stop = 60

    cuda = torch.cuda.is_available()

    encoder = MNIST_recognition_resnet(latent_dim)
    decoder = MNIST_generator_resnet(latent_dim)

    MNIST_bern_net = MNISTconv_VAE_bern_net(latent_dim, encoder, decoder, lr, cuda=cuda)

    vlb_train, vlb_dev = train_VAE(MNIST_bern_net, save_dir+'/resnet_VAE_%dd_MNIST' % (latent_dim) , batch_size, nb_epochs, trainset, valset,
                                   cuda=cuda, flat_ims=False, train_plot=False, early_stop=early_stop)


