import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GeneralCouplingLayer(nn.Module):
    def __init__(self, in_features, out_features, embed_features, masking):
        super(GeneralCouplingLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, embed_features),
            nn.ReLU(),
            nn.Linear(embed_features, embed_features),
            nn.ReLU(),
            nn.Linear(embed_features, embed_features),
            nn.ReLU(),
            nn.Linear(embed_features, embed_features),
            nn.ReLU(),
            nn.Linear(embed_features, out_features)
        )
        self.masking = masking
    
    def forward(self, x, reverse=False):
        B, N = x.shape
        x = x.reshape(B,N//2,2)
        if self.masking:
            on, off = x[:,:,0], x[:,:,1]
        else:
            
            off, on = x[:,:,0], x[:,:,1]
        if reverse:
            on = on - self.linear(off)
        else:
            on = on + self.linear(off)
        if self.masking:
            x = torch.stack([on, off], dim=2).reshape(B, N)
        else:
            x = torch.stack([off, on], dim=2).reshape(B, N)
        return x


class Flow(nn.Module):
    def __init__(self):
        super(Flow, self).__init__()
        self.flatten = nn.Flatten()
        self.cps = nn.ModuleList([
            GeneralCouplingLayer(392, 392, 1000, i%2==0) for i in range(5)
        ])
        s = nn.Parameter(torch.zeros((1,392*2)), requires_grad=True)
        self.register_parameter("s", s)

    def load(self, lr):
        self.opt = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)
    
    def forward(self, x):
        x = self.flatten(x)
        for cp in self.cps:
            x = cp(x)
        h = x*torch.exp(self.s)

        log_Ph = -(F.softplus(h) + F.softplus(-h))
        log_Ph = torch.sum(log_Ph, dim=1)
        log_S = torch.sum(self.s)
        log_PX = -(log_Ph + log_S).mean()

        return log_PX
    
    def gen(self, num):
        z = torch.distributions.Uniform(0., 1.).sample((num, 28*28)).to(device)
        h = torch.log(z) - torch.log(1. - z)
        h = h*torch.exp(-self.s)
        for cp in reversed(self.cps):
            h = cp(h, reverse=True)
        x = h.reshape(num,1,28,28)
        return x
    
    def one_epoch(self, x):
        self.opt.zero_grad()
        loss = self.forward(x)
        loss.backward()
        self.opt.step()
