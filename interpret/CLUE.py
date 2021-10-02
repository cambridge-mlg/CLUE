from __future__ import division, print_function
from src.utils import *
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
from src.probability import decompose_entropy_cat, decompose_std_gauss



class CLUE(BaseNet):
    """This will be a general class for CLUE, etc.
    A propper optimiser will be used instead of my manually designed one."""

    def __init__(self, VAE, BNN, original_x, uncertainty_weight, aleatoric_weight, epistemic_weight, prior_weight, distance_weight,
                 latent_L2_weight, prediction_similarity_weight,
                 lr, desired_preds=None, cond_mask=None, distance_metric=None, z_init=None, norm_MNIST=False, flatten_BNN=False,
                 regression=False, prob_BNN=True, cuda=True):

        """Option specification:
        MNIST: boolean, specifies whether to apply normalisation to VAE outputs before being passed on to the BNN"""
         # Load models
        self.VAE = VAE
        self.BNN = BNN
        self.BNN.set_mode_train(train=False)
        self.VAE.set_mode_train(train=False)

        # Objective function definition
        self.uncertainty_weight = uncertainty_weight
        self.aleatoric_weight = aleatoric_weight
        self.epistemic_weight = epistemic_weight
        self.prior_weight = prior_weight
        self.distance_weight = distance_weight
        self.distance_metric = distance_metric

        self.latent_L2_weight = latent_L2_weight
        self.prediction_similarity_weight = prediction_similarity_weight
        self.desired_preds = desired_preds
        # if self.desired_preds is not None:
        #     self.desired_preds = torch.Tensor(self.desired_preds)
            # if not regression:
            #     self.desired_preds = self.desired_preds#.type(torch.LongTensor)

        # Other CLUE parameters
        self.regression = regression
        # self.vae_sig = vae_sig  # We dont actually use this in our CLUE generative models as it performs worse
        self.flatten_BNN = flatten_BNN
        self.norm_MNIST = norm_MNIST
        self.original_x = torch.Tensor(original_x)


        self.prob_BNN = prob_BNN
        self.cuda = cuda
        if self.cuda:
            self.original_x = self.original_x.cuda()
            # self.z_init = self.z_init.cuda()
            if self.desired_preds is not None:
                self.desired_preds = self.desired_preds.cuda()
        self.cond_mask = cond_mask

        # Trainable params
        self.trainable_params = list()

        if self.VAE is None:  # this will be for ablation test: sensitivity analisys
            self.trainable_params.append(nn.Parameter(original_x))
        else:
            self.z_dim = VAE.latent_dim
            if z_init is not None:
                self.z_init = torch.Tensor(z_init)
                if cuda:
                    self.z_init = self.z_init.cuda()
                self.z = nn.Parameter(self.z_init)
                self.trainable_params.append(self.z)
            else:
                self.z_init = torch.zeros(self.z_dim).unsqueeze(0).repeat(original_x.shape[0],1)
                if self.cuda:
                    self.z_init = self.z_init.cuda()
                self.z = nn.Parameter(self.z_init)
                self.trainable_params.append(self.z)

        # Optimiser
        self.optimizer = Adam(self.trainable_params, lr=lr)
        # SGD(self.trainable_params, lr=lr, momentum=0.5, nesterov=True)

    def randomise_z_init(self, std):
        # assert (self.z.data == self.z_init).all()
        eps = torch.randn(self.z.shape).type(self.z.type())
        self.z.data = std * eps + self.z_init
        return None

    def pred_dist(self, preds):
        # We dont implement for now as we could just use VAEAC with class conditioning
        assert self.desired_preds is not None

        if self.regression:
            dist = F.mse_loss(preds, self.desired_preds, reduction='none')
        else:

            if len(self.desired_preds.shape) == 1 or self.desired_preds.shape[1] == 1:
                dist = F.nll_loss(preds, self.desired_preds, reduction='none')
            else:  # Soft cross entropy loss
                dist = -(torch.log(preds) * self.desired_preds).sum(dim=1)
        return dist

    def uncertainty_from_z(self):
        # if self.vae_sig:
        #     x = self.VAE.regenerate(self.z, grad=True).loc
        # else:
        # We dont use unflatten option because BNNs always take flattened input and unflatten doesnt support grad
        x = self.VAE.regenerate(self.z, grad=True)

        if self.flatten_BNN:
            to_BNN = x.view(x.shape[0], -1)
        else:
            to_BNN = x

        if self.norm_MNIST:
            to_BNN = MNIST_mean_std_norm(to_BNN)

        if self.prob_BNN:
            if self.regression:
                mu_vec, std_vec = self.BNN.sample_predict(to_BNN, Nsamples=0, grad=True)
                total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = decompose_std_gauss(mu_vec, std_vec)
                preds = mu_vec.mean(dim=0)
            else:
                probs = self.BNN.sample_predict(to_BNN, Nsamples=0, grad=True)
                total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = decompose_entropy_cat(probs)
                preds = probs.mean(dim=0)
        else:
            if self.regression:
                mu, std = self.BNN.predict(to_BNN, grad=True)
                total_uncertainty = std.squeeze(1)
                aleatoric_uncertainty = total_uncertainty
                epistemic_uncertainty = total_uncertainty*0
                preds = mu
            else:
                probs = self.BNN.predict(to_BNN, grad=True)
                total_uncertainty = -(probs * torch.log(probs + 1e-10)).sum(dim=1, keepdim=False)
                aleatoric_uncertainty = total_uncertainty
                epistemic_uncertainty = total_uncertainty*0
                preds = probs

        return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty, x, preds

    def get_objective(self, x, total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty, preds):
        # Put objectives together
        objective = self.uncertainty_weight * total_uncertainty + self.aleatoric_weight * aleatoric_uncertainty + \
                    self.epistemic_weight * epistemic_uncertainty

        if self.VAE is not None and self.cond_mask is None and self.prior_weight > 0:
            try:
                prior_loglike = self.VAE.prior.log_prob(self.z).sum(dim=1)
            except:  # This mode is just for CondCLUE but the objective method is inherited
                prior_loglike = self.VAEAC.get_prior(self.original_x, self.cond_mask, flatten=False).log_prob(self.z).sum(dim=1)
            objective += self.prior_weight * prior_loglike

        if self.latent_L2_weight != 0 and self.latent_L2_weight is not None:
            # print('latent_L2_weight is not implement correctly as distances are computed across batches')
            latent_dist = F.mse_loss(self.z, self.z_init, reduction='none').view(x.shape[0], -1).sum(dim=1)
            objective += self.latent_L2_weight * latent_dist

        if self.desired_preds is not None:
            pred_dist = self.pred_dist(preds).view(preds.shape[0], -1).sum(dim=1)
            objective += self.prediction_similarity_weight * pred_dist

        if self.distance_metric is not None:
            dist = self.distance_metric(x, self.original_x).view(x.shape[0], -1).sum(dim=1)
            objective += self.distance_weight * dist

            return objective, self.distance_weight*dist
        else:
            return objective, 0

    def optimise(self, min_steps=3, max_steps=25,
                 n_early_stop=3):
        # Vectors to capture changes for this minibatch
        z_vec = [self.z.data.cpu().numpy()]
        x_vec = []
        uncertainty_vec = np.zeros((max_steps, self.z.shape[0]))
        aleatoric_vec = np.zeros((max_steps, self.z.shape[0]))
        epistemic_vec = np.zeros((max_steps, self.z.shape[0]))
        dist_vec = np.zeros((max_steps, self.z.shape[0]))
        cost_vec = np.zeros((max_steps, self.z.shape[0]))  # this one doesnt consider the prior

        it_mask = np.zeros(self.z.shape[0])

        for step_idx in range(max_steps):

            self.optimizer.zero_grad()
            total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty, x, preds = self.uncertainty_from_z()
            objective, w_dist = self.get_objective(x, total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty, preds)
            # We sum over features and over batch size in order to make dz invariant of batch (used to average over batch size)
            objective.sum(dim=0).backward()  # backpropagate

            self.optimizer.step()

            # save vectors
            uncertainty_vec[step_idx, :] = total_uncertainty.data.cpu().numpy()
            aleatoric_vec[step_idx, :] = aleatoric_uncertainty.data.cpu().numpy()
            epistemic_vec[step_idx, :] = epistemic_uncertainty.data.cpu().numpy()
            dist_vec[step_idx, :] = (w_dist.data.cpu().numpy())
            cost_vec[step_idx, :] = (objective.data.cpu().numpy())
            x_vec.append(x.data)  # we dont convert to numpy yet because we need x0 for L1
            z_vec.append(self.z.data.cpu().numpy())  # this one is after gradient update while x is before

            it_mask = CLUE.update_stopvec(cost_vec, it_mask, step_idx, n_early_stop, min_steps)

        #  Generate final (or resulting s sample)

        x = self.VAE.regenerate(self.z, grad=False).data
        x_vec.append(x)
        x_vec = [i.cpu().numpy() for i in x_vec]  # convert x to numpy
        x_vec = np.stack(x_vec)
        z_vec = np.stack(z_vec)

        # Recover correct indexes using mask
        uncertainty_vec, epistemic_vec, aleatoric_vec, dist_vec, cost_vec, z_vec, x_vec = CLUE.apply_stopvec(it_mask,
                      uncertainty_vec, epistemic_vec, aleatoric_vec, dist_vec, cost_vec, z_vec, x_vec,
                      n_early_stop)
        return z_vec, x_vec, uncertainty_vec, epistemic_vec, aleatoric_vec, cost_vec, dist_vec

    @staticmethod
    def update_stopvec(cost_vec, it_mask, step_idx, n_early_stop, min_steps):
        # TODO: can go in a CLUE parent class
        asymptotic_rel = np.abs(cost_vec[step_idx - n_early_stop, :] - cost_vec[step_idx, :]) < cost_vec[0, :] * 1e-2
        asymptotic_abs = np.abs(cost_vec[step_idx - n_early_stop, :] - cost_vec[step_idx, :]) < 1e-3

        if step_idx > min_steps:
            condition_sum = asymptotic_rel + asymptotic_abs
        else:
            condition_sum = np.array([0])

        stop_vec = condition_sum.clip(max=1, min=0)

        to_mask = (it_mask == 0).astype(int) * stop_vec
        it_mask[to_mask == 1] = step_idx

        if (it_mask == 0).sum() == 0 and n_early_stop > 0:
            print('it %d, all conditions met, stopping' % step_idx)
        return it_mask

    @staticmethod
    def apply_stopvec(it_mask, uncertainty_vec, epistemic_vec, aleatoric_vec, dist_vec, cost_vec, z_vec, x_vec, n_early_stop):
        # uncertainty_vec[step_idx, batch_size]
        it_mask = (it_mask - n_early_stop + 1).astype(int)
        for i in range(uncertainty_vec.shape[1]):
            if it_mask[i] > 0 and n_early_stop > 0:
                uncertainty_vec[it_mask[i]:, i] = uncertainty_vec[it_mask[i], i]
                epistemic_vec[it_mask[i]:, i] = epistemic_vec[it_mask[i], i]
                aleatoric_vec[it_mask[i]:, i] = aleatoric_vec[it_mask[i], i]
                cost_vec[it_mask[i]:, i] = cost_vec[it_mask[i], i]
                dist_vec[it_mask[i]:, i] = dist_vec[it_mask[i], i]
                z_vec[it_mask[i]:, i] = z_vec[it_mask[i], i]
                x_vec[it_mask[i]:, i] = x_vec[it_mask[i], i]
        return uncertainty_vec, epistemic_vec, aleatoric_vec, dist_vec, cost_vec, z_vec, x_vec


    def sample_explanations(self, n_explanations, init_std=0.15, min_steps=3, max_steps=25,
                                    n_early_stop=3):
    # This creates a new first axis and stacks outputs there
        full_x_vec = []
        full_z_vec = []
        full_uncertainty_vec = []
        full_aleatoric_vec = []
        full_epistemic_vec = []
        full_dist_vec = []
        full_cost_vec = []

        for i in range(n_explanations):

            self.randomise_z_init(std=init_std)

            torch.autograd.set_detect_anomaly(False)

            # clue_instance.optimizer = SGD(self.trainable_params, lr=lr, momentum=0.5, nesterov=True)
            z_vec, x_vec, uncertainty_vec, epistemic_vec, aleatoric_vec, cost_vec, dist_vec = self.optimise(
                                                                    min_steps=min_steps, max_steps=max_steps,
                                                                    n_early_stop=n_early_stop)

            full_x_vec.append(x_vec)
            full_z_vec.append(z_vec)
            full_uncertainty_vec.append(uncertainty_vec)
            full_aleatoric_vec.append(aleatoric_vec)
            full_epistemic_vec.append(epistemic_vec)
            full_dist_vec.append(dist_vec)
            full_cost_vec.append(cost_vec)

        full_x_vec = np.concatenate(np.expand_dims(full_x_vec, axis=0), axis=0)
        full_z_vec = np.concatenate(np.expand_dims(full_z_vec, axis=0), axis=0)
        full_cost_vec = np.concatenate(np.expand_dims(full_cost_vec, axis=0), axis=0)
        full_dist_vec = np.concatenate(np.expand_dims(full_dist_vec, axis=0), axis=0)
        full_uncertainty_vec = np.concatenate(np.expand_dims(full_uncertainty_vec, axis=0), axis=0)
        full_aleatoric_vec = np.concatenate(np.expand_dims(full_aleatoric_vec, axis=0), axis=0)
        full_epistemic_vec = np.concatenate(np.expand_dims(full_epistemic_vec, axis=0), axis=0)

        return full_x_vec, full_z_vec, full_uncertainty_vec, full_aleatoric_vec, full_epistemic_vec, full_dist_vec, full_cost_vec

    @classmethod
    def batch_optimise(cls, VAE, BNN, original_x, uncertainty_weight, aleatoric_weight, epistemic_weight, prior_weight,
                       distance_weight, latent_L2_weight, prediction_similarity_weight, lr, min_steps=3, max_steps=25,
                       n_early_stop=3, batch_size=256, cond_mask=None, desired_preds=None,
                       distance_metric=None, z_init=None, norm_MNIST=False, flatten_BNN=False, regression=False,
                       prob_BNN=True, cuda=True):
    # This stacks outputs along the first (batch_size) axis
        full_x_vec = []
        full_z_vec = []
        full_uncertainty_vec = []
        full_aleatoric_vec = []
        full_epistemic_vec = []
        full_dist_vec = []
        full_cost_vec = []

        idx_iterator = generate_ind_batch(original_x.shape[0], batch_size=batch_size, random=False, roundup=True)
        for train_idx in idx_iterator:

            if z_init is not None:
                z_init_use = z_init[train_idx]
            else:
                z_init_use = z_init

            if desired_preds is not None:
                desired_preds_use = desired_preds[train_idx].data
            else:
                desired_preds_use = desired_preds

            CLUE_runner = cls(VAE, BNN, original_x[train_idx], uncertainty_weight, aleatoric_weight, epistemic_weight, prior_weight, distance_weight,
                              latent_L2_weight, prediction_similarity_weight, lr, cond_mask=cond_mask, distance_metric=distance_metric,
                              z_init=z_init_use, norm_MNIST=norm_MNIST, desired_preds=desired_preds_use,
                              flatten_BNN=flatten_BNN, regression=regression, prob_BNN=prob_BNN, cuda=cuda)

            z_vec, x_vec, uncertainty_vec, epistemic_vec, aleatoric_vec, cost_vec, dist_vec = \
                CLUE_runner.optimise(min_steps=min_steps, max_steps=max_steps, n_early_stop=n_early_stop)

            full_x_vec.append(x_vec)
            full_z_vec.append(z_vec)
            full_uncertainty_vec.append(uncertainty_vec)
            full_aleatoric_vec.append(aleatoric_vec)
            full_epistemic_vec.append(epistemic_vec)
            full_dist_vec.append(dist_vec)
            full_cost_vec.append(cost_vec)

        full_x_vec = np.concatenate(full_x_vec, axis=1)
        full_z_vec = np.concatenate(full_z_vec, axis=1)
        full_cost_vec = np.concatenate(full_cost_vec, axis=1)
        full_dist_vec = np.concatenate(full_dist_vec, axis=1)
        full_uncertainty_vec = np.concatenate(full_uncertainty_vec, axis=1)
        full_aleatoric_vec = np.concatenate(full_aleatoric_vec, axis=1)
        full_epistemic_vec = np.concatenate(full_epistemic_vec, axis=1)

        return full_x_vec, full_z_vec, full_uncertainty_vec, full_aleatoric_vec, full_epistemic_vec, full_dist_vec, full_cost_vec


class conditional_CLUE(CLUE):

    def __init__(self, VAEAC, BNN, original_x, uncertainty_weight, aleatoric_weight, epistemic_weight, prior_weight, distance_weight,
                 lr,  cond_mask=None, distance_metric=None, z_init=None, norm_MNIST=False, flatten_BNN=False,
                 regression=False, cuda=True):

        super(conditional_CLUE, self).__init__(VAEAC, BNN, original_x, uncertainty_weight, aleatoric_weight, epistemic_weight,
                                               prior_weight, distance_weight,
                                               lr,  cond_mask, distance_metric, z_init, norm_MNIST, flatten_BNN,
                                               regression, cuda)
        self.cond_mask = cond_mask.type(original_x.type())
        self.VAEAC = VAEAC
        self.prior_weight = 0

    def uncertainty_from_z(self):

        x = self.VAEAC.regenerate(self.z, grad=True)
        x = x * self.cond_mask + self.original_x * (1 - self.cond_mask)

        if self.flatten_BNN:
            to_BNN = x.view(x.shape[0], -1)
        else:
            to_BNN = x

        if self.norm_MNIST:
            to_BNN = MNIST_mean_std_norm(to_BNN)

        if self.regression:
            mu_vec, std_vec = self.BNN.sample_predict(to_BNN, Nsamples=0, grad=True)
            total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = decompose_std_gauss(mu_vec, std_vec)
        else:
            probs = self.BNN.sample_predict(to_BNN, Nsamples=0, grad=True)
            total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = decompose_entropy_cat(probs)

        return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty, x

    def optimise(self, min_steps=3, max_steps=25,
                 n_early_stop=3):
        # Vectors to capture changes for this minibatch
        z_vec = [self.z.data.cpu().numpy()]
        x_vec = []
        uncertainty_vec = np.zeros((max_steps, self.z.shape[0]))
        aleatoric_vec = np.zeros((max_steps, self.z.shape[0]))
        epistemic_vec = np.zeros((max_steps, self.z.shape[0]))
        dist_vec = np.zeros((max_steps, self.z.shape[0]))
        cost_vec = np.zeros((max_steps, self.z.shape[0]))  # this one doesnt consider the prior

        it_mask = np.zeros(self.z.shape[0])

        for step_idx in range(max_steps):
            self.optimizer.zero_grad()
            total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty, x = self.uncertainty_from_z()
            objective, w_dist = self.get_objective(x, total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty)
            objective.mean(dim=0).backward()  # backpropagate

            self.optimizer.step()

            # save vectors
            uncertainty_vec[step_idx, :] = total_uncertainty.data.cpu().numpy()
            aleatoric_vec[step_idx, :] = aleatoric_uncertainty.data.cpu().numpy()
            epistemic_vec[step_idx, :] = epistemic_uncertainty.data.cpu().numpy()
            dist_vec[step_idx, :] = (w_dist.data.cpu().numpy())
            cost_vec[step_idx, :] = (objective.data.cpu().numpy())
            x_vec.append(x.data)  # we dont convert to numpy yet because we need x0 for L1
            z_vec.append(self.z.data.cpu().numpy())  # this one is after gradient update while x is before

            it_mask = CLUE.update_stopvec(cost_vec, it_mask, step_idx, n_early_stop, min_steps)

        #  Generate final (or resulting s sample)

        x = self.VAE.regenerate(self.z, grad=False).data
        x = x * self.cond_mask + self.original_x * (1 - self.cond_mask)
        x_vec.append(x)
        x_vec = [i.cpu().numpy() for i in x_vec]  # convert x to numpy
        x_vec = np.stack(x_vec)
        z_vec = np.stack(z_vec)

        # Recover correct indexes using mask
        uncertainty_vec, epistemic_vec, aleatoric_vec, dist_vec, cost_vec, z_vec, x_vec = \
            CLUE.apply_stopvec(it_mask, uncertainty_vec, epistemic_vec, aleatoric_vec, dist_vec, cost_vec, z_vec,
                               x_vec, n_early_stop)
        return z_vec, x_vec, uncertainty_vec, epistemic_vec, aleatoric_vec, cost_vec, dist_vec


