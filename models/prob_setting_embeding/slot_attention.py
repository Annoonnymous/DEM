import torch
import torch.nn as nn
import torch.nn.functional as F
'''
input batch*k*in_dim features
output batch*n*out_dim slot_features
'''
class slot_attn(nn.Module):
    def __init__(self, batch_size, in_num_features, out_num_features, in_dim, out_dim, num_iter, device):
        super().__init__()
        # self.batch_size = batch_size
        self.in_num_features = in_num_features
        self.out_num_features = out_num_features
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num_iter = num_iter
        self.device = device
        self.layernorm0 = nn.LayerNorm([self.in_num_features, self.in_dim]) # inputs
        self.layernorm1 = nn.LayerNorm([self.out_num_features, self.out_dim]) #slots
        self.layernorm2 = nn.LayerNorm([self.out_num_features, self.out_dim]) #slots
        self.q_proj = nn.ModuleList([])
        self.k_proj = nn.ModuleList([])
        self.v_proj = nn.ModuleList([])
        self.mlp = nn.ModuleList([])
        for iter in range(self.num_iter):
            self.q_proj.append(nn.Linear(self.out_dim, self.out_dim))
            self.k_proj.append(nn.Linear(self.in_dim, self.out_dim))
            self.v_proj.append(nn.Linear(self.in_dim, self.out_dim))
            self.mlp.append(MLP(input_size=self.out_dim, hidden_size=512, output_size=out_dim))

        self.mu = nn.Parameter(torch.randn(1, 1, self.out_dim, device=self.device))
        self.logsigma = nn.Parameter(torch.randn(1, 1, self.out_dim, device=self.device))


    def sample_slot(self, batch_size):
        slots_init = torch.randn((batch_size, self.out_num_features, self.out_dim), device=self.device)
        samples = self.mu + self.logsigma.exp() * slots_init

        # print(samples)
        return samples
    
    def attn(self, q, k, v, iter, eps=0.0001):
        '''
        q: slots: bkd
        k: inputs: bnd
        v: inputs: bnd
        '''
        q = self.q_proj[iter](q)
        k = self.k_proj[iter](k)
        v = self.v_proj[iter](v)
        dist = torch.einsum('bkd,bnd->bkn', [q, k]) / (self.out_num_features)
        attns = dist.softmax(dim=1)

        weights = attns / (attns.sum(dim=-1, keepdim=True) + 1e-7)
        updates = torch.einsum('bkn,bnd->bkd', [weights, v])
        return updates


    def forward(self, inputs):
        slots = self.sample_slot(inputs.shape[0])
        inputs = self.layernorm0(inputs)
        for iter in range(self.num_iter):
            slots = self.layernorm1(slots)
            slots = self.attn(slots, inputs, inputs, iter=iter)
            slots = slots + self.mlp[iter](self.layernorm2(slots))
        return slots



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
