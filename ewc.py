import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ray

@ray.remote
class EWC:
    def __init__(self, model, dataloader, importance=1000, is_critic = False):
        self.model = model
        self.dataloader = dataloader
        self.importance = importance
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        self.is_critic = is_critic

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)

        self.model.eval()
        for batch in self.dataloader:
            state, action, reward, next_state, done = batch
            state = state.to(next(self.model.parameters()).device)
            action = action.to(next(self.model.parameters()).device)
            reward = reward.to(next(self.model.parameters()).device)
            next_state = next_state.to(next(self.model.parameters()).device)
            done = done.to(next(self.model.parameters()).device)

            self.model.zero_grad()

            if self.is_critic == True:
                output = self.model(state, action)
            else:
                output = self.model(state)
                
            loss = F.mse_loss(output, reward)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataloader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._means:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss