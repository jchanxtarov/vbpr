# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

import random
from typing import List, Tuple

import numpy as np

from .types import UserItems


def generate_batch_cf(
    dict_user_items: UserItems, batch_size: int, n_items: int
) -> Tuple[List[int], List[int], List[int]]:
    users_exist = list(dict_user_items.keys())
    if batch_size <= len(users_exist):
        users_batch = random.sample(users_exist, batch_size)
    else:
        users_batch = [random.choice(users_exist) for _ in range(batch_size)]

    items_pos_batch, items_neg_batch = [], []
    for u in users_batch:
        items_pos_batch += sample_items_pos_for_u(
            dict_user_items, u, 1)
        items_neg_batch += sample_items_neg_for_u(
            dict_user_items, u, 1, n_items)
    return users_batch, items_pos_batch, items_neg_batch


def sample_items_pos_for_u(
    dict_user_items: UserItems, user: int, n_sample_items_pos: int
) -> List[int]:
    items_pos = list(map(lambda x: x, dict_user_items[user]))
    n_items_pos = len(items_pos)
    items_pos_sampled: List[int] = []
    # consider: not need while & n_sample_items_pos
    while True:
        if len(items_pos_sampled) == n_sample_items_pos:
            break
        pos_item_idx = np.random.randint(low=0, high=n_items_pos, size=1)[0]
        item_pos = items_pos[pos_item_idx]
        if item_pos not in items_pos_sampled:
            items_pos_sampled.append(item_pos)
    return items_pos_sampled


def sample_items_neg_for_u(
    dict_user_items: UserItems, user: int, n_sample_items_neg: int, n_items: int
) -> List[int]:
    items_pos = list(map(lambda x: x, dict_user_items[user]))
    items_neg_sampled: List[int] = []
    while True:
        if len(items_neg_sampled) == n_sample_items_neg:
            break
        item_neg = np.random.randint(low=0, high=n_items, size=1)[0]
        if (item_neg not in items_pos) and (item_neg not in items_neg_sampled):
            items_neg_sampled.append(item_neg)
    return items_neg_sampled
