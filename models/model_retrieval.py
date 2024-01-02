import torch
from models import DEM, load_pretrained, AllGather
import torch.nn as nn
import torch.nn.functional as F


class DEM_Retrieval(DEM):
    def __init__(self, config, device):
        super().__init__(config, load_vision_params=config['load_params'], load_text_params=config['load_params'],
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=config['mlm'],device=device)

        if not self.pa100k_only_img_classifier:
            self.mlm = config['mlm']
            self.pa100k = config['pa100k']
            self.total_epoch = config['schedular']['epochs']
            self.diffusion = config['diffusion']
            self.soft_margin = config['soft_margin']
            self.gmm = config['gmm']
            self.slot = config['slot']
            self.eda = config['eda']

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("vision_encoder missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, attr_text_ids=None, attr_text_atts=None, attr_text_ids_masked=None,
                attr_masked_pos=None, attr_masked_ids=None, label=None, text_ids_eda=None, text_atts_eda=None, cur_epoch=None,
                gpt_input=None):
        add_loss = dict()
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        image_feat, text_feat = self.get_features(image_embeds, text_embeds)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                          text_embeds, text_atts, text_feat, idx=idx)
        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos,
                                        masked_ids)
        add_loss.update({'loss_mlm': loss_mlm})
        # eda
        if self.eda:
            text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
            text_feat_eda = self.get_features(text_embeds=text_embeds_eda)
            loss_itc_eda = self.get_contrastive_loss(image_feat, text_feat_eda, idx=idx)
            loss_itm_eda = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                                  text_embeds_eda, text_atts_eda, text_feat_eda, idx=idx)
            loss_itc = loss_itc + 0.8 * loss_itc_eda
            loss_itm = loss_itm + 0.8 * loss_itm_eda

        add_loss.update({'loss_itc': loss_itc})
        add_loss.update({'loss_itm': loss_itm})

        # loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)
        # add_loss.update({'loss_mlm': loss_mlm})

        if self.diffusion:
            with torch.no_grad():
                gpt_embeds = self.get_text_embeds(gpt_input.input_ids, gpt_input.attention_mask)
            gpt_embeds = gpt_embeds.detach()
            detach_embeds = text_embeds.detach()
            loss_diffusion = self.ddpm_sampler.loss(detach_embeds.unsqueeze(1), None, gpt_embeds.unsqueeze(1))
            add_loss.update({'loss_diffusion': loss_diffusion})
        
        if self.soft_margin:
            gpt_ids = gpt_input.input_ids
            gpt_atts = gpt_input.attention_mask
            gpt_embed = self.get_text_embeds(gpt_ids, gpt_atts)
            gpt_feat = self.text_proj(gpt_embed)[:, 0, :]
            dist_text = torch.einsum('be,be->b', [gpt_feat, text_feat])
            loss_soft = self.get_soft_match_loss(dist_text, 0.3, image_feat, text_feat) * 0.0

            loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                                    gpt_embed, gpt_atts, gpt_feat, idx=idx)
            # another implementation
            mlp1out = self.soft_mlp1(image_feat)
            mlp2out = self.soft_mlp2(image_feat)
            fc_out = self.soft_fc(image_feat)
            # IA_IG = gpt_feat @ mlp1out.t()
            # IA_IQ = text_feat @ mlp1out.t()
            IA_IG = mlp1out @ gpt_feat.t()
            IA_IQ = mlp1out @ text_feat.t()
            SGQ = F.cosine_similarity(IA_IG, IA_IQ)

            # EA_IQ = text_feat @ mlp2out.t()
            # Tr = text_feat @ fc_out.t()
            EA_IQ = mlp2out @ text_feat.t()
            Tr = fc_out @ text_feat.t()
            EM = F.cosine_similarity(EA_IQ, Tr)

            tri_sim = SGQ + EM

            loss_bbc = torch.exp(tri_sim) / torch.sum(torch.exp(tri_sim), dim=-1)
            # print(loss_bbc)
            loss_bbc = torch.sum(torch.log(loss_bbc), dim=-1) / mlp1out.shape[0]
            # print(loss_bbc)
            add_loss.update({'loss_soft': loss_soft - loss_bbc + loss_itm})
        if self.gmm:
            img_mu, img_1alpha, img_beta = self.image_prob_con(image_feat)
            txt_mu, txt_1alpha, txt_beta = self.text_prob_con(text_feat)
            loss_i = self.get_prob_loss(img_mu, img_1alpha, img_beta, image_feat, 1., 5e-2)
            loss_t = self.get_prob_loss(txt_mu, txt_1alpha, txt_beta, text_feat, 1., 5e-2)
            loss_i4t = self.get_prob_loss(img_mu, img_1alpha, img_beta, text_feat, 1., 5e-2)
            loss_t4i = self.get_prob_loss(txt_mu, txt_1alpha, txt_beta, image_feat, 1., 5e-2)
            loss_gmm = loss_i + loss_t + 1e-4 * (loss_i4t + loss_t4i)
            add_loss.update({'loss_gmm': loss_gmm})
        if self.slot:
            s1, s2 = self.agm(image_embeds.detach(), text_embeds.detach(), image_feat.detach(), text_feat.detach())
            loss_slot = self.get_slot_loss(s1, s2) + self.get_slot_triplet_loss(s1, s2)
            add_loss.update({'loss_slot': loss_slot})
            
        return add_loss
