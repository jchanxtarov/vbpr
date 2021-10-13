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

        self.embed_user = nn.Embedding(dataset.n_users, dim_embed_global)
        nn.init.xavier_uniform_(self.embed_user.weight,
                                gain=nn.init.calculate_gain('relu'))
        self.embed_item = nn.Embedding(dataset.n_items, dim_embed_global)
        nn.init.xavier_uniform_(self.embed_item.weight,
                                gain=nn.init.calculate_gain('relu'))

        self.rate_reg = rate_reg

    def forward(self, *input):
        return self._compute_loss(*input)

    def _compute_loss(self, users, items_pos, items_neg):
        embed_user = self.embed_user(users)  # (batch_size, dim_embed_global)
        # (batch_size, dim_embed_global)
        embed_item_pos = self.embed_item(items_pos)
        # (batch_size, dim_embed_global)
        embed_item_neg = self.embed_item(items_neg)

        score_pos = th.sum(embed_user * embed_item_pos, dim=1)  # (batch_size)
        score_neg = th.sum(embed_user * embed_item_neg, dim=1)  # (batch_size)

        base_loss = (-1.0) * F.logsigmoid(score_pos - score_neg)
        base_loss = th.mean(base_loss)

        reg_loss = self._l2_loss(
            embed_user) + self._l2_loss(embed_item_pos) + self._l2_loss(embed_item_neg)
        reg_loss = self.rate_reg * reg_loss
        return base_loss, reg_loss

    def _l2_loss(self, embedd):
        return th.sum(embedd.pow(2).sum(1) / 2.0)

    def predict(self, users, items):
        # (n_eval_users, dim_embed_global)
        embed_user = self.embed_user(users)
        # (n_eval_items, dim_embed_global)
        embed_item = self.embed_item(items)
        # (n_eval_users, n_eval_items)
        return th.matmul(embed_user, embed_item.transpose(0, 1))
