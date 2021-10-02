from __future__ import division, print_function
import numpy as np
import torch
import torch.nn.functional as F


def evaluate_NN_net_cat(net, valloader):

    test_cost = 0  # Note that these are per sample
    test_err = 0
    nb_samples = 0

    for j, (x, y) in enumerate(valloader):
        y = y.cuda().type(torch.long)
        x = x.view(x.shape[0], -1)
        loss, err, probs = net.eval(x, y)
        test_cost += loss.item()
        test_err += err.cpu().numpy()
        nb_samples += len(x)

    # test_cost /= nb_samples
    test_err /= nb_samples
    print('Loglike = %6.6f, err = %1.6f\n' % (-test_cost, test_err))
    return -test_cost, test_err


def evaluate_NN_net_gauss(net, valloader, y_means, y_stds):
    mu_vec = []
    sigma_vec = []
    y_vec = []

    for x,y in valloader:
        mu, sig = net.predict(x)
        mu_vec.append(mu.data.cpu())
        sigma_vec.append(sig.data.cpu())
        y_vec.append(y.data.cpu())

    mu_vec = torch.cat(mu_vec)
    sigma_vec = torch.cat(sigma_vec)
    y_vec = torch.cat(y_vec)

    rms, ll = net.unnormalised_eval(mu_vec, sigma_vec, y_vec, y_mu=y_means, y_std=y_stds)
    print('rms', rms, 'll', ll)

    return ll, rms