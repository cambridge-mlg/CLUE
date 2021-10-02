from __future__ import division
import torch
from src.probability import diagonal_gauss_loglike, get_rms, get_loglike
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from src.utils import BaseNet, to_variable, cprint
from src.radam import RAdam


class NN_cat(BaseNet):  # for categorical distributions
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, cuda=True):
        super(NN_cat, self).__init__()

        cprint('y', 'NN categorical output')
        self.lr = lr
        self.model = model
        self.cuda = cuda
        self.weight_decay = weight_decay

        self.create_net()
        self.create_opt()
        self.schedule = None  # [] #[50,200,400,600]
        self.epoch = 0

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)
        if self.cuda:
            self.model.cuda()
            cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def fit(self, x, y):
        self.set_mode_train(train=True)
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = F.cross_entropy(out, y, reduction='mean')
        loss.backward()

        self.optimizer.step()

        # out: (batch_size, out_channels, out_caps_dims)
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data * x.shape[0], err

    def eval(self, x, y):
        self.set_mode_train(train=False)
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model(x)
        loss = F.cross_entropy(out, y, reduction='sum')
        probs = F.softmax(out, dim=1).data.cpu()

        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def predict(self, x, grad=False):
        self.set_mode_train(train=False)
        x, = to_variable(var=(x, ), cuda=self.cuda)
        if grad:
            self.optimizer.zero_grad()
            if not x.requires_grad:
                x.requires_grad = True
        out = self.model(x)
        probs = F.softmax(out, dim=1)
        if grad:
            return probs
        else:
            return probs.data


class NN_gauss(BaseNet):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, cuda=True, eps=1e-3):
        super(NN_gauss, self).__init__()
        cprint('y', ' Creating Net!! ')
        cprint('y', 'BNN gaussian output')
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = model
        self.cuda = cuda
        self.eps=eps

        self.create_net()
        self.create_opt()
        self.schedule = None  # [] #[50,200,400,600]
        self.epoch = 0

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        if self.cuda:
            self.model.cuda()
            cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def fit(self, x, y):
        self.set_mode_train(train=True)
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()
        mu, sigma = self.model(x)
        sigma = sigma.clamp(min=self.eps)
        loss = -diagonal_gauss_loglike(y, mu, sigma).mean(dim=0)

        loss.backward()

        self.optimizer.step()

        return loss.data * x.shape[0], mu.data, sigma.data

    def eval(self, x, y):
        self.set_mode_train(train=False)
        x, y = to_variable(var=(x, y), cuda=self.cuda)
        mu, sigma = self.model(x)
        sigma = sigma.clamp(min=self.eps)
        loss = -diagonal_gauss_loglike(y, mu, sigma).mean(dim=0)

        return loss.data * x.shape[0], mu.data, sigma.data

    @staticmethod
    def unnormalised_eval(pred_mu, pred_std, y, y_mu, y_std):
        rms = get_rms(pred_mu, y, y_mu, y_std)  # this already computes sum
        ll = get_loglike(pred_mu, pred_std, y, y_mu, y_std)  # this already computes sum
        return rms, ll

    def predict(self, x, grad=False):
        self.set_mode_train(train=False)
        x, = to_variable(var=(x,), cuda=self.cuda)
        if grad:
            self.optimizer.zero_grad()
            if not x.requires_grad:
                x.requires_grad = True
        mu, sigma = self.model(x)
        if grad:
            return mu, sigma
        else:
            return mu.data, sigma.data
