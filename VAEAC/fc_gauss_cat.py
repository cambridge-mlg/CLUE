from __future__ import division
import torch
import numpy as np
from src.layers import SkipConnection
from src.utils import BaseNet, to_variable, cprint
from src.probability import normal_parse_params
from src.radam import RAdam

from torch.distributions import kl_divergence
import torch.backends.cudnn as cudnn
from src.gauss_cat import *
from .models import MLP_prior_net, MLP_recognition_net, MLP_generator_net, \
    MLP_preact_prior_net, MLP_preact_recognition_net, MLP_preact_generator_net


class VAEAC_gauss_cat(nn.Module):
    def __init__(self, input_dim_vec, width, depth, latent_dim, pred_sig=True):
        super(VAEAC_gauss_cat, self).__init__()

        self.input_dim = 0
        self.input_dim_vec = input_dim_vec
        for e in self.input_dim_vec:
            self.input_dim += e

        self.latent_dim = latent_dim
        self.recognition_net = MLP_preact_recognition_net(self.input_dim, width, depth, latent_dim)
        self.prior_net = MLP_preact_prior_net(self.input_dim, width, depth, latent_dim)
        if pred_sig:
            raise Exception('Not yet implemented')
        else:
            self.generator_net = MLP_preact_generator_net(self.input_dim, width, depth, latent_dim)
            self.rec_loglike = rms_cat_loglike(self.input_dim_vec, reduction='none')
        self.pred_sig = pred_sig
        self.sigma_mu = 1e4
        self.sigma_sigma = 1e-4


    @staticmethod
    def apply_mask(x, mask):
        """Positive bits in mask are set to 0 in x (observed)"""
        observed = x.clone()  # torch.tensor(x)
        observed[mask.bool()] = 0
        return observed

    def recognition_encode(self, x):
        """Works with flattened representATION"""
        approx_post_params = self.recognition_net(x)
        approx_post = normal_parse_params(approx_post_params, 1e-3)
        return approx_post

    def prior_encode(self, x, mask):
        """Works with flattened representATION"""
        x = self.apply_mask(x, mask)
        x = torch.cat([x, mask], 1)
        prior_params = self.prior_net(x)
        prior = normal_parse_params(prior_params, 1e-3)
        return prior

    def decode(self, z_sample):
        rec_params = self.generator_net(z_sample)
        return rec_params

    def reg_cost(self, prior):
        num_objects = prior.mean.shape[0]
        mu = prior.mean.view(num_objects, -1)
        sigma = prior.scale.view(num_objects, -1)
        mu_regularizer = -(mu ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
        sigma_regularizer = (sigma.log() - sigma).sum(-1) * self.sigma_sigma
        return mu_regularizer + sigma_regularizer

    def vlb(self, prior, approx_post, x, rec_params):
        if self.pred_sig:
            raise Exception('Not yet imeplemented')
        else:
            rec = self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
        prior_regularization = self.reg_cost(prior).view(x.shape[0], -1).sum(-1)
        kl = kl_divergence(approx_post, prior).view(x.shape[0], -1).sum(-1)
        return rec - kl + prior_regularization

    def iwlb(self, prior, approx_post, x, K=50):
        estimates = []
        for i in range(K):
            latent = approx_post.rsample()
            rec_params = self.decode(latent)
            if self.pred_sig:
                raise Exception('Not yet imeplemented')
            else:
                rec_loglike = self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)

            prior_log_prob = prior.log_prob(latent)
            prior_log_prob = prior_log_prob.view(x.shape[0], -1)
            prior_log_prob = prior_log_prob.sum(-1)

            proposal_log_prob = approx_post.log_prob(latent)
            proposal_log_prob = proposal_log_prob.view(x.shape[0], -1)
            proposal_log_prob = proposal_log_prob.sum(-1)

            estimate = rec_loglike + prior_log_prob - proposal_log_prob
            estimates.append(estimate[:, None])

        return torch.logsumexp(torch.cat(estimates, 1), 1) - np.log(K)


class VAEAC_gauss_cat_net(BaseNet):
    def __init__(self, input_dim_vec, width, depth, latent_dim, pred_sig=False, lr=1e-3, cuda=True, flatten=True):
        super(VAEAC_gauss_cat_net, self).__init__()
        cprint('y', 'VAE_gauss_net')

        self.cuda = cuda

        self.input_dim = 0
        self.input_dim_vec = input_dim_vec
        for e in self.input_dim_vec:
            self.input_dim += e
        self.flatten = flatten
        if not self.flatten:
            pass
            # raise Exception('Error calculation not supported without flattening')

        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.lr = lr
        self.pred_sig = pred_sig

        self.create_net()
        self.create_opt()
        self.epoch = 0
        self.schedule = None

        self.vlb_scale = 1 / len(self.input_dim_vec)  # scale for dimensions of input so we can use same LR always

    def create_net(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.model = VAEAC_gauss_cat(self.input_dim_vec, self.width, self.depth, self.latent_dim, self.pred_sig)
        if self.cuda:
            self.model = self.model.cuda()
            cudnn.benchmark = True
        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = RAdam(self.model.parameters(), lr=self.lr)  # torch.optim.Adam

    def fit(self, x, mask):
        self.set_mode_train(train=True)

        if self.flatten:
            mask = gauss_cat_to_flat_mask(mask, self.input_dim_vec)
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)

        x_flat, x, mask = to_variable(var=(x_flat, x, mask), cuda=self.cuda)
        self.optimizer.zero_grad()

        prior = self.model.prior_encode(x_flat, mask)
        approx_post = self.model.recognition_encode(x_flat)
        z_sample = approx_post.rsample()
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(prior, approx_post, x, rec_params)
        loss = (- vlb * self.vlb_scale).mean()

        loss.backward()
        self.optimizer.step()

        if self.pred_sig:
            rec_return = normal_parse_params(rec_params, 1e-3)
        else:
            rec_return = rec_params
        return vlb.mean().item(), rec_return

    def eval(self, x, mask, sample=False):
        self.set_mode_train(train=False)

        if self.flatten:
            mask = gauss_cat_to_flat_mask(mask, self.input_dim_vec)
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)

        x_flat, x, mask = to_variable(var=(x_flat, x, mask), cuda=self.cuda)
        prior = self.model.prior_encode(x_flat, mask)

        approx_post = self.model.recognition_encode(x_flat)
        if sample:
            z_sample = approx_post.sample()
        else:
            z_sample = approx_post.loc
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(prior, approx_post, x, rec_params)

        if self.pred_sig:
            rec_return = normal_parse_params(rec_params, 1e-3)
        else:
            rec_return = rec_params
        return vlb.mean().item(), rec_return

    def eval_iw(self, x, mask, k=50):
        self.set_mode_train(train=False)

        if self.flatten:
            mask = gauss_cat_to_flat_mask(mask, self.input_dim_vec)
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)


        x_flat, x, mask = to_variable(var=(x_flat, x, mask), cuda=self.cuda)

        prior = self.model.prior_encode(x, mask)
        approx_post = self.model.recognition_encode(x_flat)

        iw_lb = self.model.iwlb(prior, approx_post, x, k)
        return iw_lb.mean().item()

    def get_prior(self, x, mask, flatten=None):
        self.set_mode_train(train=False)
        if flatten is None:
            flatten = self.flatten
        if flatten:
            mask = gauss_cat_to_flat_mask(mask, self.input_dim_vec)
            x = gauss_cat_to_flat(x, self.input_dim_vec)

        x, mask = to_variable(var=(x, mask), cuda=self.cuda)
        prior = self.model.prior_encode(x, mask)
        return prior

    def get_post(self, x, flatten=None):
        self.set_mode_train(train=False)
        if flatten is None:
            flatten = self.flatten
        if flatten:
            x = gauss_cat_to_flat(x, self.input_dim_vec)

        x, = to_variable(var=(x,), cuda=self.cuda)
        approx_post = self.model.recognition_encode(x)
        return approx_post

    def inpaint(self, x, mask, Nsample=1, z_mean=False, flatten=False, unflatten=False, cat_probs=False):
        self.set_mode_train(train=False)

        if flatten:
            mask = gauss_cat_to_flat_mask(mask, self.input_dim_vec)
            x = gauss_cat_to_flat(x, self.input_dim_vec)

        x, mask = to_variable(var=(x, mask), cuda=self.cuda)
        prior = self.model.prior_encode(x, mask)
        out = []
        for i in range(Nsample):
            if z_mean:
                z_sample = prior.loc.data
            else:
                z_sample = prior.sample()
            rec_params = self.model.decode(z_sample)
            out.append(rec_params.data)
        out = torch.stack(out, dim=0)

        if self.pred_sig:
            raise Exception('Not yet implemented')
        else:
            dim0 = out.shape[0]
            dim1 = out.shape[1]
            out = out.view(dim0*dim1, -1)
            if unflatten:
                out = flat_to_gauss_cat(out, self.input_dim_vec)
            else:
                out = selective_softmax(out, self.input_dim_vec, grad=False, cat_probs=cat_probs)
            out = out.view(dim0, dim1, -1)
            return out.data

    def regenerate(self, z, grad=False, unflatten=False):
        self.set_mode_train(train=False)
        if unflatten and grad:
            raise Exception('flatten and grad options are not compatible')
        if grad:
            if not z.requires_grad:
                z.requires_grad = True
        else:
            z, = to_variable(var=(z,), volatile=True, cuda=self.cuda)
        out = self.model.decode(z)
        if self.pred_sig:
            raise Exception('Not yet implemented')
        else:
            if unflatten:
                out = flat_to_gauss_cat(out, self.input_dim_vec)
            else:
                out = selective_softmax(out, self.input_dim_vec, grad=grad)
            return out.data
