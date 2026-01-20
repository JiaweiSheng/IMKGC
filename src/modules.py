import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import pdb

class Similarity(nn.Module):
    """
    cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.sim_func = Similarity(temp=temp)

    def forward(self, emb1, emb2):
        batch_sim = self.sim_func(emb1.unsqueeze(1), emb2.unsqueeze(0))
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels)

class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, z_dim, args):
        super(MuSigmaEncoder, self).__init__()
        self.r_dim = r_dim
        self.z_dim = z_dim

        self.args = args
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.hidden_to_mu.weight)
        nn.init.xavier_uniform_(self.hidden_to_sigma.weight)
        nn.init.zeros_(self.hidden_to_mu.bias)
        nn.init.zeros_(self.hidden_to_sigma.bias)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = r
        mu = self.hidden_to_mu(hidden)
        sigma = F.softplus(self.hidden_to_sigma(hidden))
        sigma = torch.clamp(sigma, max=self.args.clamp) 
        return torch.distributions.Normal(mu, sigma)
