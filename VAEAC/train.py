from __future__ import print_function
from __future__ import division
import torch, time
import torch.utils.data
import matplotlib.pyplot as plt

from torchvision.utils import save_image, make_grid
from src.utils import *
from numpy.random import normal

# TODO: implement image masks

def train_VAEAC(net, masker, name, batch_size, nb_epochs, trainset, valset, cuda,
                flat_ims=False, train_plot=False, Nclass=None, early_stop=None, script_mode=False):

    models_dir = name + '_models'
    results_dir = name + '_results'
    mkdir(models_dir)
    mkdir(results_dir)

    if cuda:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=3)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                num_workers=3)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                                  num_workers=3)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                num_workers=3)

## ---------------------------------------------------------------------------------------------------------------------
# net dims
    cprint('c', '\nNetwork:')

    epoch = 0

    ## ---------------------------------------------------------------------------------------------------------------------
    # train
    cprint('c', '\nTrain:')

    print('  init cost variables:')
    vlb_train = np.zeros(nb_epochs)
    vlb_dev = np.zeros(nb_epochs)
    iwlb_dev = np.zeros(nb_epochs)
    best_vlb = -np.inf
    best_epoch = 0

    nb_its_dev = 1

    tic0 = time.time()
    for i in range(epoch, nb_epochs):
        net.set_mode_train(True)

        tic = time.time()
        nb_samples = 0
        for x, y in trainloader:

            if flat_ims:
                x = x.view(x.shape[0], -1)
            if Nclass is not None:
                y_oh = torch_onehot(y, Nclass).type(x.type())
                x = torch.cat([x, y_oh], 1)

            mask = masker(x)
            cost, _ = net.fit(x, mask)

            vlb_train[i] += cost * len(x)
            nb_samples += len(x)

        vlb_train[i] /= nb_samples

        toc = time.time()

        # ---- print
        print("it %d/%d, vlb %f, " % (i, nb_epochs, vlb_train[i]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))
        net.update_lr(i)

        # ---- dev
        if i % nb_its_dev == 0:
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):

                if flat_ims:
                    x = x.view(x.shape[0], -1)
                if Nclass is not None:
                    y_oh = torch_onehot(y, Nclass).type(x.type())
                    x = torch.cat([x, y_oh], 1)

                mask = masker(x)
                cost, rec_mean = net.eval(x, mask)
                # iwlb = net.eval_iw(x, mask, 25)

                vlb_dev[i] += cost * len(x)
                # iwlb_dev[i] += iwlb
                nb_samples += len(x)

            vlb_dev[i] /= nb_samples
            # iwlb_dev[i] /= nb_samples

            cprint('g', '    vlb %f (%f)\n' % (vlb_dev[i], best_vlb))

            if train_plot:
                xm = net.model.apply_mask(x, mask)
                
                xr = x.cpu()
                rec_inpaint = net.inpaint(xm, mask)
                try:
                    o = rec_mean.cpu()
                    rec_inpaint = rec_inpaint[0].cpu()
                except:
                    o = rec_mean.loc.cpu()
                    rec_inpaint = rec_inpaint[0].loc.cpu()

                if Nclass is not None:
                    xm = xm[:, :-Nclass]
                    rec_inpaint = rec_inpaint[:, :-Nclass]
                    xr = xr[:, :-Nclass]
                    o = o[:, :-Nclass]

                if len(x.shape) == 2:
                    side = int(np.sqrt(xm.shape[1]))
                    xm = xm.view(-1, 1, side, side).data
                    rec_inpaint = rec_inpaint.view(-1, 1, side, side).data
                    xr = xr.view(-1, 1, side, side).data
                    o = o.view(-1, 1, side, side).data

                import matplotlib.pyplot as plt
                plt.figure()
                dd = make_grid(torch.cat([xr[:10], o[:10]]), nrow=10).numpy()
                plt.imshow(np.transpose(dd, (1, 2, 0)), interpolation='nearest')
                plt.title('reconstruct')
                if script_mode:
                    plt.savefig(results_dir + '/rec%d.png' % i)
                else:
                    plt.show()

                import matplotlib.pyplot as plt
                plt.figure()
                dd = make_grid(torch.cat([xm[:10], rec_inpaint[:10]]), nrow=10).numpy()
                plt.imshow(np.transpose(dd, (1, 2, 0)), interpolation='nearest')
                plt.title('inpaint')
                if script_mode:
                    plt.savefig(results_dir + '/inp%d.png' % i)
                else:
                    plt.show()

        if vlb_dev[i] > best_vlb:
            best_vlb = vlb_dev[i]
            best_epoch = i
            net.save(models_dir + '/theta_best.dat')

        if early_stop is not None and (i - best_epoch) > early_stop:
            break


    net.save(models_dir + '/theta_last.dat')
    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    ## ---------------------------------------------------------------------------------------------------------------------
    # results
    cprint('c', '\nRESULTS:')
    nb_parameters = net.get_nb_parameters()
    best_cost_dev = np.max(vlb_dev)
    # best_iw_dev = np.max(iwlb_dev)
    best_cost_train = np.max(vlb_train)

    print('  best_vlb_dev: %f' % best_cost_dev)
    # print('  best_iwlb_dev: %f' % best_iw_dev)
    print('  best_vlb_train: %f' % best_cost_train)
    print('  nb_parameters: %d (%s)\n' % (nb_parameters, humansize(nb_parameters)))

    ## ---------------------------------------------------------------------------------------------------------------------
    # fig cost vs its
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(np.clip(vlb_train, -1000, 1000), 'r')
    plt.plot(np.clip(vlb_dev[::nb_its_dev], -1000, 1000), 'b')
    plt.legend(['cost_train', 'cost_dev'])
    plt.ylabel('vlb')
    plt.xlabel('it')
    plt.grid(True)
    plt.savefig(results_dir+'/train_cost.png')

    # plt.figure()
    # plt.plot(np.clip(iwlb_dev[::nb_its_dev], -1000, 1000), 'b')
    # plt.ylabel('dev iwlb')
    # plt.xlabel('it')
    # plt.grid(True)
    # plt.savefig(results_dir + '/train_iwlb.png')
    if train_plot:
        plt.show()
    return vlb_train, vlb_dev




