# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

import logging
import pickle
from dataclasses import dataclass

import torch as th

from .base import BaseDataset


# TODO: load image & update parameters in cnn model
@dataclass
class VbprDataset(BaseDataset):
    dim_imgfeat: int
    items_imgfeat: th.Tensor

    def __init__(self, dataset: str, is_sample_dataset: bool = False) -> None:
        super().__init__(dataset=dataset)

        self.items_imgfeat = self._load_imgfeat(
            f"./src/datasets/{dataset}/itemid2imgfeat.pkl", is_sample_dataset)
        self.dim_imgfeat = len(self.items_imgfeat[0])

    def _load_imgfeat(self, path_dataset: str, is_sample_dataset: bool = False) -> th.Tensor:
        if is_sample_dataset:
            return th.randn(self.n_items, 4096)  # f (n_items, dim_imgfeat)

        with open(path_dataset, mode='rb') as f:
            itemid2imgfeat = pickle.load(f)

        items_imgfeat = th.FloatTensor(
            [itemid2imgfeat[idx].detach().numpy() for idx in range(self.n_items)])
        return items_imgfeat

    def logging_statistics(self) -> None:
        logging.info("n_train: {}".format(self.n_train))
        logging.info("n_test:  {}".format(self.n_test))
        logging.info("n_users: {}".format(self.n_users))
        logging.info("n_items: {}".format(self.n_items))
        logging.info("dim_imgfeat: {}".format(self.dim_imgfeat))
