import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class ODEFunc(nn.Module):

    def __init__(self, feature_dim, temporal_dim):
        super(ODEFunc, self).__init__()
        self.adj = None
        self.x0 = None
        # self.alpha = nn.Parameter(0.8 * torch.ones(self.adj.shape[1]))
        self.alpha = None
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)

    def forward(self, t, x):
        # self.alpha = nn.Parameter(0.8 * torch.ones(self.adj.shape[1]))

        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).cuda()
        xa = torch.einsum('ij, kjlm->kilm', self.adj, x)

        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, lm->ijkm', x, w)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        f = (alpha / 2 * xa - x + xw - x + xw2 - x + self.x0)
        return f


class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def set_adj_alpha(self, adj):
        self.odefunc.adj = adj
        # if self.odefunc.alpha==None:
            # self.odefunc.alpha = nn.Parameter(0.8 * torch.ones(self.odefunc.adj.shape[1]))
        self.odefunc.alpha = 0.8 * torch.ones(self.odefunc.adj.shape[1])

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t,method='euler')[1] # torchdiffeq框架
        return z


# Define the ODEGCN model.
class ODEG(nn.Module): # adj(1600,1600)
    def __init__(self, feature_dim, temporal_dim, time):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim), t=torch.tensor([0, time]))

    def forward(self, x, adj): # x.size(1600,64)
        self.odeblock.set_x0(x)
        self.odeblock.set_adj_alpha(adj)
        z = self.odeblock(x)
        return F.relu(z)
