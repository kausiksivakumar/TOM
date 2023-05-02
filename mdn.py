import os
import csv
from collections import OrderedDict
import itertools

from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# import matplotlib.pyplot as plt

from utils import *

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class _LinearBlock(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim):
        super().__init__(OrderedDict([
            ("fc", torch.nn.Linear(input_dim, output_dim)),
            #             ("norm", torch.nn.BatchNorm1d(output_dim)),
            ("swish", Swish()),
        ]))



class MDN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(200, 200, 200, 200), mixture_size=10, lr=0.001):
        super().__init__()

        self.mixture_size = mixture_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Feature extractor
        hidden_layers = []
        prev_dims = [input_dim] + list(hidden_dims[:-1])
        for i, (prev_dim, current_dim) in enumerate(zip(prev_dims, hidden_dims)):
            hidden_layers.append((f"hidden{i + 1}", _LinearBlock(prev_dim, current_dim)))
        self.hidden_layers = torch.nn.Sequential(OrderedDict(hidden_layers))

        # Final output layer: mean and variance
        self.final_layer = torch.nn.Linear(hidden_dims[-1], 3 * mixture_size * output_dim)

        # Parameters for the features / noise
        self.register_parameter("_noise", torch.nn.Parameter(torch.zeros(1)))

        self.inputs_mu = nn.Parameter(torch.zeros(1, self.input_dim), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(1, self.input_dim), requires_grad=False)

        self.targets_mu = nn.Parameter(torch.zeros(1, self.output_dim), requires_grad=False)
        self.targets_sigma = nn.Parameter(torch.zeros(1, self.output_dim), requires_grad=False)
        self.fit_input = False

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # self.max_logvar = nn.Parameter(torch.ones(self.output_dim, mixture_size, dtype=torch.float32) / 2.0)
        # self.min_logvar = nn.Parameter(-torch.ones(self.output_dim, mixture_size, dtype=torch.float32) * 10.0)

    def forward(self, inputs):
        # # Transform inputs
        # if self.fit_input:
        #     inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        features = self.hidden_layers(inputs)

        mean, logvar, weight_logits = self.final_layer(features).reshape(
            (inputs.shape[0], 3, self.output_dim, self.mixture_size)).transpose(0, 1)

        variance = F.softplus(logvar) + torch.nn.functional.softplus(self._noise)
        mixture_distribution = torch.distributions.Categorical(logits=weight_logits)
        component_distribution = torch.distributions.Normal(mean, variance.sqrt())
        output = torch.distributions.MixtureSameFamily(mixture_distribution, component_distribution)
        return output

    def _save_best(self, epoch, holdout_loss):
        updated = False

        current = holdout_loss
        _, best = self._snapshot
        improvement = (best-current) / abs(best)
        if improvement > 0.01:
            self._snapshot = (epoch, current)
            updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def fit_input_stats(self, inputs, targets, device='cuda'):
        self.fit_input = True
        inputs_mu = np.mean(inputs, axis=0, keepdims=True)
        inputs_sigma = np.std(inputs, axis=0, keepdims=True)
        inputs_sigma[inputs_sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(inputs_mu).to(device).float()
        self.inputs_sigma.data = torch.from_numpy(inputs_sigma).to(device).float()

        targets_mu = np.mean(targets, axis=0, keepdims=True)
        targets_sigma = np.std(targets, axis=0, keepdims=True)
        targets_sigma[targets_sigma < 1e-12] = 1.0

        self.targets_mu.data = torch.from_numpy(targets_mu).to(device).float()
        self.targets_sigma.data = torch.from_numpy(targets_sigma).to(device).float()

    def loss(self, output, target,weights = None):
        # If there are no weights, then its MBPO - weight all equally
        if(weights is None):
            weights     =   1
        losses  =   -output.log_prob(target)
        losses  =   losses*weights   
        return losses.mean()

    def predict(self, inputs, batch_size=50000, device='cuda:0'):
        ''' sample generation. '''
        ensemble_samples = np.zeros((1, inputs.shape[0], self.output_dim))
        with torch.no_grad():
            for i in range(0, inputs.shape[0], batch_size):
                input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
                input = (input - self.inputs_mu) / self.inputs_sigma
                output = self(input)
                samples = output.sample()
                samples = self.targets_mu + self.targets_sigma * samples
                ensemble_samples[:, i:min(i + batch_size, inputs.shape[0]), :] = samples.detach().cpu().numpy()

        return ensemble_samples

    def train(self, inputs, targets, batch_size=256, holdout_ratio=0.2,weights = None,
              max_logging=5000, max_epochs_since_update=5, max_epochs=50,
              device='cuda:0'):

        # TODO:Get weights for TOM sampling
        if(weights is None):
            weights                 =   np.ones((inputs.shape[0],1))
        #Clipping weights to 10
        weights                     =   weights.clip(None,10)
        self._max_epochs_since_update = max_epochs_since_update
        self._snapshot = (None, 1e10) # keeping track of the best val epoch
        self._epochs_since_update = 0

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[idxs]

        # num_holdout = int(inputs.shape[0] * holdout_ratio)
        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]][:10000]
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]][:10000]
        
        # normalization using the training set
        self.fit_input_stats(inputs, targets)

        input_val   = torch.from_numpy(holdout_inputs).float().to(device)
        target_val  = torch.from_numpy(holdout_targets).float().to(device)
        input_val   = (input_val - self.inputs_mu) / self.inputs_sigma
        target_val  = (target_val - self.targets_mu) / self.targets_sigma

        train_prob,holdout_weights  =   weights[permutation[num_holdout:]], weights[permutation[:num_holdout]]
        weights_val                 =   torch.from_numpy(holdout_weights[:10000]).float().to(device)

        train_prob                  =   (train_prob/train_prob.sum()).flatten()

        # idxs                        =   np.random.randint(inputs.shape[0], size=[inputs.shape[0]])
        # idxs                        =   np.random.choice(inputs.shape[0], size=[100*batch_size*max_epochs], p= train_prob)

        if max_epochs is not None:
            epoch_iter = range(max_epochs)
        else:
            epoch_iter = itertools.count()

        grad_update = 0
        for epoch in tqdm(epoch_iter):
            idxs                        =   np.random.choice(inputs.shape[0], size=[100*batch_size], p= train_prob)
            for batch_num in range(0,100*batch_size,batch_size):
                # batch_idxs = idxs[batch_num * batch_size:(batch_num + 1) * batch_size]
                batch_idxs  =   idxs[batch_num:batch_num+batch_size]
                input       =   torch.from_numpy(inputs[batch_idxs]).float().to(device)
                target      =   torch.from_numpy(targets[batch_idxs]).float().to(device)

                if self.fit_input:
                    input = (input - self.inputs_mu) / self.inputs_sigma
                    target = (target - self.targets_mu) / self.targets_sigma

                pred_distribution = self(input)
                loss = self.loss(pred_distribution, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                grad_update += 1

            idxs = shuffle_rows(idxs)
            # Since val might contain 0 elements
            if(len(input_val)==0):
                rmse    =   input[0][0]
                val_loss=   input[0][0]
                continue

            with torch.no_grad():
                pred_distribution = self(input_val)
                rmse = torch.sqrt(((pred_distribution.mean - target_val) ** 2).mean())
                val_loss = self.loss(pred_distribution, target_val,weights=weights_val)

                break_train = self._save_best(epoch, val_loss)

            if break_train:
                break
            # print(f"Epoch {epoch} Train {loss.cpu().detach().item():.3f} Val {val_loss.cpu().detach().item():.3f}")

        # Just to keep consistent with ensemble API
        self.elite_model_idxes = [0]

        return rmse.cpu().detach().item(), val_loss.cpu().detach().item()

    # def predict(self, loader, device='cuda:0'):
    #     rmses, nlls = [], []
    #
    #     for x_batch, y_batch in tqdm(loader):
    #         input = x_batch.to(device)
    #         target = y_batch.to(device)
    #
    #         pred_dist = self(input)
    #         nll = -pred_dist.log_prob(target).mean(dim=-1)
    #         rmse = torch.sqrt(((pred_dist.mean - target) ** 2).mean(dim=-1))
    #
    #         rmses.append(rmse.detach().cpu())
    #         nlls.append(nll.detach().cpu())
    #
    #     return torch.cat(rmses, dim=-1), torch.cat(nlls, dim=-1)
    #
    # def train_model(self, train_loader, test_loader, epochs=400, device='cuda:0'):
    #     for epoch in range(epochs):
    #         if epoch % 1 == 0:
    #             with torch.no_grad():
    #                 rmses, nlls = self.predict(test_loader, device=device)
    #                 print(f"Testing Epoch {epoch} \t RMSE {rmses.mean().item():.2f} \t NLL {nlls.mean().item():.2f}")
    #
    #         for x_batch, y_batch in tqdm(train_loader):
    #             input = x_batch.to(device)
    #             target = y_batch.to(device)
    #
    #             pred_distribution = self(input)
    #             loss = self.loss(pred_distribution, target)
    #
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

#             with torch.no_grad():
#                 rmses, nlls = self.predict(train_loader, device=device)
#                 print(f"Training Epoch {epoch} \t RMSE {rmses.mean().item():.2f} \t NLL {nlls.mean().item():.2f}")

if __name__ == "__main__":
    mdn = MDN(5,4)
    mdn.to('cuda')
    x = torch.randn((512, 5)).numpy()
    y = torch.randn((512, 4)).numpy()
    mdn.train(x, y)
    samples = mdn.predict(x)
