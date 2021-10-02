from __future__ import division, print_function 
import sys
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % int(sys.argv[1])


dname = str(sys.argv[2])
print(dname)


N_up = 2
nb_dir = '/'.join(os.getcwd().split('/')[:-N_up])
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

names = ['wine', 'default_credit', 'compas', 'lsat']
widths = [300, 300, 300, 300] # [200, 200, 200, 200]
depths = [3, 3, 3, 3] # We go deeper because we are using residual models
latent_dims = [6, 8, 4, 4]

from src.compas_loader import get_my_COMPAS, X_dims_to_input_dim_vec
from src.LSAT_loader import get_my_LSAT

if dname == 'wine':
    from src.UCI_loader import load_UCI
    import torch
    from torchvision import datasets, transforms
    from VAE.fc_gauss import VAE_gauss_net
    from VAE.train import train_VAE
    from src.utils import Datafeed

    for latent_dim in [2, 3, 4, 5, 6, 8, 12, 16]:

        print(dname)

        x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds = \
        load_UCI(dset_name=dname, splits=10, seed=42, separate_targets=True, save_dir='../data/')

        trainset = Datafeed(x_train, x_train, transform=None)
        valset = Datafeed(x_test, x_train, transform=None)

        save_dir = ('../saves/fc_preact_VAE_d%d_' % latent_dim)  + dname

        input_dim = x_train.shape[1]
        width = widths[names.index(dname)]
        depth = depths[names.index(dname)] # number of hidden layers
        # latent_dim = latent_dims[names.index(dname)]

        batch_size = 128
        nb_epochs = 2500
        early_stop = 200
        lr = 1e-4

        cuda = torch.cuda.is_available()

        net = VAE_gauss_net(input_dim, width, depth, latent_dim, pred_sig=False, lr=lr, cuda=cuda)

        vlb_train, vlb_dev = train_VAE(net, save_dir, batch_size, nb_epochs, trainset, valset,
                                    cuda=cuda, flat_ims=False, train_plot=False, early_stop=early_stop)

if dname == 'default_credit':
    import torch
    from torchvision import datasets, transforms
    from VAE.fc_gauss_cat import VAE_gauss_cat_net
    from VAE.train import train_VAE
    from src.utils import Datafeed

    from src.UCI_loader import load_UCI, unnormalise_cat_vars
    import numpy as np

    x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds = \
        load_UCI(dset_name='default_credit', splits=10, seed=42, separate_targets=True, save_dir='../data/')
    input_dim_vec = [1, 2, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    x_train = unnormalise_cat_vars(x_train, x_means, x_stds, input_dim_vec)
    x_test = unnormalise_cat_vars(x_test, x_means, x_stds, input_dim_vec)
    # target unnormalisation
    y_train = unnormalise_cat_vars(y_train, y_means, y_stds, [2])
    y_test = unnormalise_cat_vars(y_test, y_means, y_stds, [2])

    for latent_dim in [2, 3, 4, 5, 6, 8, 12, 16]:

        trainset = Datafeed(x_train, x_train, transform=None)
        valset = Datafeed(x_test, x_test, transform=None)

        save_dir = ('../saves/fc_preact_VAE_d%d_' % latent_dim) + dname

        width = widths[names.index(dname)]
        depth = depths[names.index(dname)]  # number of hidden layers
        # latent_dim = latent_dims[names.index(dname)]

        batch_size = 128
        nb_epochs = 2500
        lr = 1e-4
        early_stop = 200

        cuda = torch.cuda.is_available()

        net = VAE_gauss_cat_net(input_dim_vec, width, depth, latent_dim, pred_sig=False, lr=lr, cuda=cuda, flatten=True)

        vlb_train, vlb_dev = train_VAE(net, save_dir, batch_size, nb_epochs, trainset, valset,
                                       cuda=cuda, flat_ims=False, train_plot=False, early_stop=early_stop)


if dname == 'compas':

    import torch
    from torchvision import datasets, transforms
    from VAE.fc_gauss_cat import VAE_gauss_cat_net
    from VAE.train import train_VAE
    from src.utils import Datafeed

    from src.UCI_loader import load_UCI, unnormalise_cat_vars
    import numpy as np

    x_train, x_test, x_means, x_stds, y_train, y_test, feature_names, X_dims = \
        get_my_COMPAS(rseed=42, separate_test=True, test_ratio=0.1, save_dir='../data/')
    input_dim_vec = X_dims_to_input_dim_vec(X_dims)
    print('Compas', x_train.shape, x_test.shape)
    print(input_dim_vec)

    for latent_dim in [2, 3, 4, 5, 6, 8, 12, 16]:
        print(dname)

        trainset = Datafeed(x_train, x_train, transform=None)
        valset = Datafeed(x_test, x_test, transform=None)

        save_dir = ('../saves/fc_preact_VAE_d%d_' % latent_dim) + dname

        width = widths[names.index(dname)]
        depth = depths[names.index(dname)]  # number of hidden layers

        batch_size = 128
        nb_epochs = 2500
        lr = 1e-4
        early_stop = 200

        cuda = torch.cuda.is_available()

        net = VAE_gauss_cat_net(input_dim_vec, width, depth, latent_dim, pred_sig=False, lr=lr, cuda=cuda, flatten=False)

        vlb_train, vlb_dev = train_VAE(net, save_dir, batch_size, nb_epochs, trainset, valset,
                                       cuda=cuda, flat_ims=False, train_plot=False, early_stop=early_stop)

if dname == 'lsat':
    import torch
    from torchvision import datasets, transforms
    from VAE.fc_gauss_cat import VAE_gauss_cat_net
    from VAE.train import train_VAE
    from src.utils import Datafeed

    from src.UCI_loader import load_UCI, unnormalise_cat_vars
    import numpy as np

    x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds, my_data_keys, input_dim_vec = \
        get_my_LSAT(save_dir='../data/')
    print('LSAT', x_train.shape, x_test.shape)
    print(input_dim_vec)

    for latent_dim in [2, 3, 4, 5, 6, 8, 12, 16]:
        trainset = Datafeed(x_train, x_train, transform=None)
        valset = Datafeed(x_test, x_test, transform=None)

        save_dir = ('../saves/fc_preact_VAE_d%d_' % latent_dim) + dname

        width = widths[names.index(dname)]
        depth = depths[names.index(dname)]  # number of hidden layers
        # latent_dim = latent_dims[names.index(dname)]

        batch_size = 128
        nb_epochs = 2500
        lr = 1e-4
        early_stop = 200

        cuda = torch.cuda.is_available()

        net = VAE_gauss_cat_net(input_dim_vec, width, depth, latent_dim, pred_sig=False, lr=lr, cuda=cuda, flatten=False)

        vlb_train, vlb_dev = train_VAE(net, save_dir, batch_size, nb_epochs, trainset, valset,
                                       cuda=cuda, flat_ims=False, train_plot=False, early_stop=early_stop)


