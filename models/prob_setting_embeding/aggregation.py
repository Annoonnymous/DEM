import torch
import torch.nn as nn
import torch.nn.functional as F
from .slot_attention import slot_attn

class aggre_module(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.batch_size = config['batch_size_train']
        self.num_features = 3
        self.text_aggre = slot_attn(batch_size=self.batch_size, in_num_features=56, 
                                    out_num_features=self.num_features, in_dim=768, 
                                    out_dim=256, num_iter=3, device=device)
        self.image_aggre = slot_attn(batch_size=self.batch_size, in_num_features=49, 
                                    out_num_features=self.num_features, in_dim=1024, 
                                    out_dim=256, num_iter=3, device=device)
        self.layernorm1 = nn.LayerNorm([256])
        self.layernorm2 = nn.LayerNorm([3, 256])
        self.t2i_proj = nn.Linear(3, 3)
        self.i2t_proj = nn.Linear(3, 3)

        
    def forward(self, image_embeds, text_embeds, image_feats, text_feats):
        bsi = image_embeds.shape[0]
        bst = text_embeds.shape[0]
        image_slots = self.layernorm2(self.image_aggre(image_embeds)
                                      ) + self.layernorm1(image_feats).unsqueeze(1).repeat(1, self.num_features, 1)
        text_slots = self.layernorm2(self.text_aggre(text_embeds)
                                     ) + self.layernorm1(text_feats).unsqueeze(1).repeat(1, self.num_features, 1) 
        # [batch_size, 3, embed_dim]

        sim1 = torch.einsum('bnd,xyd->bxny', [text_slots, image_slots]) # batch, batch, 3, 3
        sim1 = F.layer_norm(sim1.reshape(bst * bsi, self.num_features, self.num_features), 
                            [self.num_features, self.num_features]).view(
                                bst, bsi, self.num_features, self.num_features
                            ).contiguous()
        sim1 = self.t2i_proj(sim1)
        sim1 = torch.sum(torch.log(torch.exp(sim1).sum(dim=-1)), dim=-1) / (2. * 3.)

        sim2 = torch.einsum('bnd,xyd->bxny', [image_slots, text_slots])
        sim2 = F.layer_norm(sim2.reshape(bsi * bst, self.num_features, self.num_features), 
                            [self.num_features, self.num_features]).view(
                                bsi, bst, self.num_features, self.num_features
                            ).contiguous()
        sim2 = self.i2t_proj(sim2)
        sim2 = torch.sum(torch.log(torch.exp(sim2).sum(dim=-1)), dim=-1) / (2. * 3.)
        return sim1, sim2





        
        


