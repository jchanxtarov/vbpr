# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

# torch.utils.data.Dataset-dependent implementation

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ..types import UserItems


class VbprDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name_dataset,
        ext='.jpg',
        transform=None,
    ):
        self.ext = ext
        self.transform = transform
        self.image_dir = f'./src/datasets/{name_dataset}/images/'

        path_train_cf = f'./src/datasets/{name_dataset}/train.txt'
        path_test_cf = f'./src/datasets/{name_dataset}/test.txt'
        self.arr_train_pos, self.dict_train_pos = self._load_cf_txt(
            path_train_cf)
        self.arr_test_pos, self.dict_test_pos = self._load_cf_txt(
            path_test_cf)

        # get statics
        self.uniq_users = sorted(list(self.dict_train_pos.keys()))
        self.n_users = max(self.uniq_users) + 1
        self.n_items = max(
            max(self.arr_train_pos[:, 1]), max(self.arr_test_pos[:, 1])
        ) + 1
        self.n_train = len(self.arr_train_pos)
        self.n_test = len(self.arr_test_pos)

    def __len__(self) -> int:
        return self.n_users

    def __getitem__(self, idx_user):
        """
        sampling
        """
        idx_items_pos = self.dict_train_pos[idx_user]
        idx = np.random.randint(low=0, high=len(idx_items_pos), size=1)[0]
        idx_item_pos = idx_items_pos[idx]

        while True:
            idx_item_neg = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if idx_item_pos != idx_item_neg:
                break

        """
        prepare batch dataset
        """
        image_pos = self.load_transform_img(idx_item_pos)
        image_neg = self.load_transform_img(idx_item_neg)

        output = {
            'users': idx_user,
            'items_pos': idx_item_pos,
            'items_neg': idx_item_neg,
            'items_image_pos': image_pos,
            'items_image_neg': image_neg,
        }
        return output

    def _load_cf_txt(self, path_dataset: str) -> Tuple[np.ndarray, UserItems]:
        dict_uid_iids = dict()
        list_uid_iid = list()

        lines = open(path_dataset, "r").readlines()
        for line in lines:
            tmps = line.strip()
            inters = [int(i) for i in tmps.split(" ")]

            user, items = inters[0], inters[1:]
            items = list(set(items))

            for item in items:
                list_uid_iid.append([user, item])

            if len(items) > 0:
                dict_uid_iids[user] = items
        return np.array(list_uid_iid), dict_uid_iids

    def load_transform_img(self, item):
        path_img = f'{self.image_dir}/{item}{self.ext}'
        image_array = Image.open(path_img).convert('RGB')
        if self.transform:
            image = self.transform(image_array)
        else:
            image = torch.Tensor(
                np.transpose(image_array, (2, 0, 1))) / 255  # for 0~1 scaling
        return image
