import torch
import torch.nn as nn
class BayesCap_MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: int, Input dimension
        out_dim: int, Output dimension
        hid_dim: int, hidden dimension
        num_layers: Number of hidden layers
        p_drop: dropout probability 
    '''
    def __init__(
        self, 
        inp_dim, 
        out_dim,
        hid_dim=512, 
        num_layers=1, 
        p_drop=0,
    ):
        super(BayesCap_MLP, self).__init__()
        mod = []
        for layer in range(num_layers):
            if layer==0:
                incoming = inp_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
            elif layer==num_layers//2:
                incoming = hid_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
                mod.append(nn.Dropout(p=p_drop))
            elif layer==num_layers-1:
                incoming = hid_dim
                outgoing = out_dim
                mod.append(nn.Linear(incoming, outgoing))
        self.mod = nn.Sequential(*mod)

        self.block_mu = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.block_alpha = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

        self.block_beta = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x_intr = self.mod(x)
        # print('dbg', x_intr.shape, x.shape)
        x_intr = x_intr + x
        x_mu = self.block_mu(x_intr)
        x_1alpha = self.block_alpha(x_intr)
        x_beta = self.block_beta(x_intr)
        return x_mu, x_1alpha, x_beta
# import torch
# import torch.nn as nn
# class BayesCap_MLP(nn.Module):
#     '''
#     Baseclass to create a simple MLP
#     Inputs
#         inp_dim: int, Input dimension
#         out_dim: int, Output dimension
#         hid_dim: int, hidden dimension
#         num_layers: Number of hidden layers
#         p_drop: dropout probability 
#     '''
#     def __init__(
#         self, 
#         inp_dim, 
#         out_dim,
#         hid_dim=512, 
#         num_layers=1, 
#         p_drop=0,
#     ):
#         super(BayesCap_MLP, self).__init__()
#         mod = []
#         for layer in range(num_layers):
#             if layer==0:
#                 mod.append(nn.Linear(inp_dim, hid_dim))
#                 mod.append(nn.ReLU())
#             elif layer==num_layers//2:
#                 mod.append(nn.Linear(hid_dim, hid_dim))
#                 mod.append(nn.ReLU())
#                 mod.append(nn.Dropout(p=p_drop))
#             elif layer==num_layers-1:
#                 mod.append(nn.Linear(hid_dim, inp_dim))
#         self.mod = nn.Sequential(*mod)

#         self.block_mu = nn.Sequential(
#             nn.Linear(inp_dim, out_dim),
#             nn.ReLU(),
#             nn.Linear(out_dim, out_dim),
#         )

#         self.block_alpha = nn.Sequential(
#             nn.Linear(inp_dim, out_dim),
#             nn.ReLU(),
#             # nn.Linear(out_dim, out_dim),
#             # nn.ReLU(),
#             nn.Linear(out_dim, out_dim),
#             nn.ReLU(),
#         )

#         self.block_beta = nn.Sequential(
#             nn.Linear(inp_dim, out_dim),
#             nn.ReLU(),
#             # nn.Linear(out_dim, out_dim),
#             # nn.ReLU(),
#             nn.Linear(out_dim, out_dim),
#             nn.ReLU(),
#         )
    
#     def forward(self, x):
#         x_intr = self.mod(x)
#         # print('dbg', x_intr.shape, x.shape)
#         x_intr = x_intr + x
#         x_mu = self.block_mu(x_intr)
#         x_1alpha = self.block_alpha(x_intr)
#         x_beta = self.block_beta(x_intr)
#         return x_mu, x_1alpha, x_beta
