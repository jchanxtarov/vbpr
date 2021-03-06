# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

from typing import List

import torch as th
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from utils.loaders.vbpr2 import VbprDataset


class VBPR(nn.Module):
    def __init__(
            self,
            dataset: VbprDataset,
            dim_embed_latent: int,  # K
            dim_embed_visual: int,  # D
            rates_reg: List[float],  # [reg_embed, reg_beta, reg_trans_e]
    ) -> None:
        super(VBPR, self).__init__()

        self.embed_user = nn.Embedding(
            dataset.n_users, dim_embed_latent)
        nn.init.xavier_uniform_(
            self.embed_user.weight, gain=nn.init.calculate_gain('relu'))
        self.embed_item = nn.Embedding(dataset.n_items, dim_embed_latent)
        nn.init.xavier_uniform_(
            self.embed_item.weight, gain=nn.init.calculate_gain('relu'))
        self.embed_user_visual = nn.Embedding(
            dataset.n_users, dim_embed_visual)
        nn.init.xavier_uniform_(
            self.embed_user.weight, gain=nn.init.calculate_gain('relu'))

        model = models.alexnet(pretrained=True)
        self.body = create_feature_extractor(
            model, {'classifier.4': 'FC7'},
        )
        _x = th.rand((1, 3, dataset.size_img, dataset.size_img))
        dim_imgfeat = self.body(_x)['FC7'].size()[1]  # (batch, dim_imgfeat)

        self.trans_e = nn.Parameter(th.Tensor(dim_embed_visual, dim_imgfeat))  # E (D, F)
        nn.init.xavier_uniform_(self.trans_e, gain=nn.init.calculate_gain('relu'))

        # NOTE: users' overall opinion toward the visual appearance of a given item
        self.bias_visual = nn.Parameter(th.Tensor(dim_imgfeat, 1))  # Beta' (F, 1)
        nn.init.xavier_uniform_(self.bias_visual, gain=nn.init.calculate_gain('relu'))

        self.rate_reg_embed = rates_reg[0]
        self.rate_reg_beta = rates_reg[1]
        self.rate_reg_trans_e = rates_reg[2]

    def forward(self, *input):
        return self._compute_loss(*input)

    def _compute_loss(self, users, items_pos, items_neg, items_image_pos, items_image_neg):
        # get each embedding
        embed_user_lat = self.embed_user(users)  # (batch_size, dim_embed_latent)
        embed_item_lat_pos = self.embed_item(items_pos)  # (batch_size, dim_embed_latent)
        embed_item_lat_neg = self.embed_item(items_neg)  # (batch_size, dim_embed_latent)

        embed_user_vis = self.embed_user_visual(users)  # (batch_size, dim_embed_visual)
        imgfeat_item_vis_pos = self.body(items_image_pos)['FC7']  # (batch_size, dim_imgfeat)
        imgfeat_item_vis_neg = self.body(items_image_neg)['FC7']  # (batch_size, dim_imgfeat)

        # compute score with latent factors
        score_lat_pos = th.bmm(
            embed_user_lat.unsqueeze(1), embed_item_lat_pos.unsqueeze(2)
        ).squeeze(2)  # (batch_size)
        score_lat_neg = th.bmm(
            embed_user_lat.unsqueeze(1), embed_item_lat_neg.unsqueeze(2)
        ).squeeze(2)  # (batch_size)

        # compute score with visual factors
        embed_item_vis_pos = th.matmul(self.trans_e, imgfeat_item_vis_pos.T).T  # (batch_size, dim_embed_visual)
        embed_item_vis_neg = th.matmul(self.trans_e, imgfeat_item_vis_neg.T).T  # (batch_size, dim_embed_visual)
        score_vis_pos = th.bmm(embed_user_vis.unsqueeze(1), embed_item_vis_pos.unsqueeze(2)).squeeze(2)  # (batch_size)
        score_vis_neg = th.bmm(embed_user_vis.unsqueeze(1), embed_item_vis_neg.unsqueeze(2)).squeeze(2)  # (batch_size)

        # compute score of users' overall opinion toward the visual appearance
        bias_vis_pos = th.matmul(imgfeat_item_vis_pos, self.bias_visual)  # (batch_size)
        bias_vis_neg = th.matmul(imgfeat_item_vis_neg, self.bias_visual)  # (batch_size)

        # Eq.(4)
        score_pos = score_lat_pos + score_vis_pos + bias_vis_pos  # (batch_size)
        score_neg = score_lat_neg + score_vis_neg + bias_vis_neg  # (batch_size)

        # Eq.(6) & Eq.(7)
        loss_base = (-1.0) * F.logsigmoid(score_pos - score_neg)
        loss_base = th.mean(loss_base)

        # Eq.(8)
        loss_reg = 0.0
        loss_reg += self.rate_reg_embed * self._l2_loss(embed_user_lat)
        loss_reg += self.rate_reg_embed * self._l2_loss(embed_item_lat_pos)
        loss_reg += self.rate_reg_embed * self._l2_loss(embed_item_lat_neg)
        loss_reg += self.rate_reg_embed * self._l2_loss(embed_user_vis)
        loss_reg += self.rate_reg_beta * self._l2_loss(self.bias_visual)
        loss_reg += self.rate_reg_trans_e * self._l2_loss(self.trans_e)

        return loss_base, loss_reg

    def _l2_loss(self, embedd):
        return th.sum(embedd.pow(2).sum(1) / 2.0)

    def predict(self, users, items, items_imgfeat):
        # get each embedding
        embed_user_lat = self.embed_user(users)  # (n_eval_users, dim_embed_latent)
        embed_item_lat = self.embed_item(items)  # (n_eval_items, dim_embed_latent)
        embed_user_vis = self.embed_user_visual(users)  # (n_eval_users, dim_embed_visual)

        # compute score with latent factors
        score_lat = th.matmul(embed_user_lat, embed_item_lat.T)  # (n_eval_users, n_eval_items)

        # compute score with visual factors
        embed_item_vis = th.matmul(self.trans_e, items_imgfeat.T).T  # (n_eval_items, dim_embed_visual)
        score_vis = th.matmul(embed_user_vis, embed_item_vis.T)  # (n_eval_users, n_eval_items)

        # compute score with visual factors
        bias_vis = th.sum(self.bias_visual.T * items_imgfeat, dim=1)  # (n_eval_items)

        # Eq.(4)
        score = score_lat + score_vis + bias_vis.unsqueeze(0)
        return score  # (n_eval_users, n_eval_items)

    def compute_item_vis_embeds(self, items_img):
        imgfeat_item_vis = self.body(items_img)['FC7']  # (n_eval_items, dim_imgfeat)
        return imgfeat_item_vis
