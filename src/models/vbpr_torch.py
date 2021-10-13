# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

from typing import List

import torch as th
from torch import nn
from torch.nn import functional as F
from utils.loaders.vbpr import VbprDataset


class VBPR(nn.Module):
    def __init__(
            self,
            dataset: VbprDataset,
            dim_embed_global: int,
            rate_reg: float
    ) -> None:
        super(VBPR, self).__init__()

        self.user_embed = nn.Embedding(dataset.n_users, dim_embed_global)
        nn.init.xavier_uniform_(self.user_embed.weight,
                                gain=nn.init.calculate_gain('relu'))
        self.item_embed = nn.Embedding(dataset.n_items, dim_embed_global)
        nn.init.xavier_uniform_(self.item_embed.weight,
                                gain=nn.init.calculate_gain('relu'))

        self.rate_reg = rate_reg

    def forward(self, *input):
        return self._compute_loss(*input)

    def _compute_loss(self, userids, itemids_pos, itemids_neg):
        user_embed = self.user_embed(userids)  # (batch_size, dim_embed_global)
        # (batch_size, dim_embed_global)
        item_embed_pos = self.item_embed(itemids_pos)
        # (batch_size, dim_embed_global)
        item_embed_neg = self.item_embed(itemids_neg)

        pos_score = th.sum(user_embed * item_embed_pos, dim=1)  # (batch_size)
        neg_score = th.sum(user_embed * item_embed_neg, dim=1)  # (batch_size)

        base_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        base_loss = th.mean(base_loss)

        reg_loss = self._l2_loss(
            user_embed) + self._l2_loss(item_embed_pos) + self._l2_loss(item_embed_neg)
        reg_loss = self.rate_reg * reg_loss
        return base_loss, reg_loss

    def _l2_loss(self, embedd):
        return th.sum(embedd.pow(2).sum(1) / 2.0)

    def predict(self, userids, itemids):
        # (n_eval_users, dim_embed_global)
        user_embed = self.user_embed(userids)
        # (n_eval_items, dim_embed_global)
        item_embed = self.item_embed(itemids)
        # (n_eval_users, n_eval_items)
        return th.matmul(user_embed, item_embed.transpose(0, 1))
