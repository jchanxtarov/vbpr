# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

import logging
from dataclasses import dataclass
from time import time
from typing import Dict, List, Tuple

import numpy as np
import torch as th
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from utils.helpers import ensure_file, generate_path_model_file, is_best_epoch
from utils.loaders.vbpr2 import VbprDataset
from utils.metrics import compute_metrics
from utils.samplers import generate_batch_cf
from utils.types import UserItems

from .base import BasePredictor
from .vbpr_torch import VBPR


@dataclass
class VbprPredictor(BasePredictor):
    # parameters
    model: VBPR
    epochs: int
    dim_embed_latent: int  # K
    dim_embed_visual: int  # D
    dim_imgfeat: int  # F
    rates_reg: List[float]  # [reg_embed, reg_beta, reg_trans_e]
    batch_size: int
    n_loop_cf: int
    rate_learning: float
    top_ks: List[int]
    interval_evaluate: int
    stopping_steps: int

    # save
    best_epoch: int

    # dataset
    dataset: VbprDataset

    def __init__(
        self,
        epochs=50,
        dim_embed_latent=64,
        dim_embed_visual=64,
        batch_size=1024,
        rates_reg=[1e-5, 1e-5, 1e-5],
        rate_learning=1e-4,
        top_ks=[20, 60, 100],
        interval_evaluate=10,
        stopping_steps=30,
    ) -> None:
        super().__init__()
        self.epochs = epochs
        self.dim_embed_latent = dim_embed_latent
        self.dim_embed_visual = dim_embed_visual
        self.rates_reg = rates_reg
        self.batch_size = batch_size
        self.rate_learning = rate_learning
        self.top_ks = top_ks
        self.interval_evaluate = interval_evaluate
        self.stopping_steps = stopping_steps
        self.best_epoch = 0

    def load(self, dataset: VbprDataset) -> None:
        self.model = VBPR(
            dataset=dataset,
            dim_embed_latent=self.dim_embed_latent,
            dim_embed_visual=self.dim_embed_visual,
            rates_reg=self.rates_reg,
        )
        self.dataset = dataset
        logging.debug('Success to load torch VBPR model.')

    def train(self, dataloader: th.utils.data.DataLoader) -> None:
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model.to(device)
        logging.info(self.model)

        # same as tensorflow: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.rate_learning, eps=1e-7
        )

        # ids
        users_batch_test = [th.LongTensor(self.dataset.uniq_users[i:i + 10000]).to(
            device) for i in range(0, self.dataset.n_users, 10000)]
        items = th.arange(self.dataset.n_items, dtype=th.long).to(device)

        # train model
        time_start_train = time()
        self.model = self.model.train()
        list_hit, list_recall, list_precision, list_ndcg = [], [], [], []

        # for epoch in tqdm(range(1, self.epochs + 1), position=0, desc="[Epoch]", leave=False, colour='green'):
        for epoch in range(1, self.epochs + 1):
            time_start_epoch = time()

            # 1. optimize recommendation (CF) part
            loss_cf, loss_reg = 0.0, 0.0

            for _, data in enumerate(tqdm(
                dataloader, position=1, desc="[Batch]", leave=False, colour='red')
            ):
                batch_users = Variable(data['users']).to(device)
                batch_items_pos = Variable(data['items_pos']).to(device)
                batch_items_neg = Variable(data['items_neg']).to(device)
                batch_items_image_pos = Variable(data['items_image_pos']).to(device)
                batch_items_image_neg = Variable(data['items_image_neg']).to(device)

                loss_cf_batch, loss_reg_batch = self.model(
                    batch_users, batch_items_pos, batch_items_neg,
                    batch_items_image_pos, batch_items_image_neg,
                )
                loss_batch = loss_cf_batch + loss_reg_batch
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
                loss_cf += loss_cf_batch.item()
                loss_reg += loss_reg_batch.item()

            logging.info(
                '   [Training] Epoch {:04d} / {:04d} [{:.1f}s] : CF(base) Loss {:.4f} : Regularization(reg) Loss {:.4f}'.format(
                    epoch, self.epochs, time() - time_start_epoch, loss_cf, loss_reg))

            # 2. evaluation
            if (epoch % self.interval_evaluate) == 0:
                time_start_eval = time()
                # TODO: add image feature by cnn
                hits, precisions, recalls, ndcgs, _ = self._evaluate_batch(
                    dict_train_pos=self.dataset.dict_train_pos,
                    dict_test_pos=self.dataset.dict_test_pos,
                    users_batch=users_batch_test,
                    items=items,
                    list_k=self.top_ks,
                )

                logging.info('[Evaluation] Epoch {:04d} / {:04d} [{:.1f}s] : hits [{:s}], precision [{:s}], recall [{:s}], ndcg [{:s}]'.format(
                    epoch, self.epochs, time() - time_start_eval,
                    '\t'.join(['%.5f' % h for h in hits]),
                    '\t'.join(['%.5f' % p for p in precisions]),
                    '\t'.join(['%.5f' % r for r in recalls]),
                    '\t'.join(['%.5f' % n for n in ndcgs]),
                ))

                # consider: which indicator should be used?
                list_hit.append(hits[0])
                list_precision.append(precisions[0])
                list_recall.append(recalls[0])
                list_ndcg.append(ndcgs[0])

                is_best, should_stop = is_best_epoch(
                    list_recall, epoch, self.interval_evaluate, self.stopping_steps)
                if is_best:
                    self.best_epoch = epoch
                    best_model = self.model
                if should_stop:
                    break

        # NOTE: to use best_model for prediction
        self.model = best_model
        logging.info('[Finish Training] Best Epoch {:04d} [{:.1f}s]'.format(
            self.best_epoch, time() - time_start_train))

    # TODO: add image feature by cnn
    def _evaluate_batch(
        self, dict_train_pos: UserItems, dict_test_pos: UserItems, users_batch: List[th.Tensor], items: th.Tensor, list_k: List[int], is_predict: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict[int, List[int]]]:
        self.model = self.model.eval()

        hit, precision, recall, ndcg = [], [], [], []
        rec_items = dict()
        n_users = 0
        with th.no_grad():
            n_batchs = (self.dataset.n_items // self.batch_size) + 1
            batchs_items = [items[i*self.batch_size:(i+1)*self.batch_size] for i in range(n_batchs)]

            _items_imgfeat = []
            for batch_items in batchs_items:
                items_img = th.Tensor([self.dataset.load_transform_img(item).tolist() for item in batch_items])
                _items_imgfeat.append(self.model.compute_item_vis_embeds(items_img))
            items_imgfeat = th.cat(_items_imgfeat, dim=0)  # (n_items, dim_imgfeat)

            for users in users_batch:
                cf_scores_batch = self.model.predict(
                    users, items, items_imgfeat)  # (n_batch_users, n_eval_items)

                # TODO: add image features
                hits_batch, precision_batch, recall_batch, ndcg_batch, rec_items_batch = compute_metrics(
                    cf_scores_batch.cpu(), dict_train_pos, dict_test_pos,
                    users.cpu().numpy(), items.cpu().numpy(), list_k, is_predict)

                hit.append(hits_batch)
                precision.append(precision_batch)
                recall.append(recall_batch)
                ndcg.append(ndcg_batch)
                n_users += len(users)

                if is_predict:
                    for i, user in enumerate(users):
                        rec_items[int(user)] = list(rec_items_batch[i])

        hits, precisions, recalls, ngcds = [], [], [], []
        for i in range(len(list_k)):
            hits.append(np.sum([h[i] for h in hit]) / n_users)
            precisions.append(np.sum([p[i] for p in precision]) / n_users)
            recalls.append(np.sum([r[i] for r in recall]) / n_users)
            ngcds.append(np.sum([n[i] for n in ndcg]) / n_users)

        return hits, precisions, recalls, ngcds, rec_items

    def save(self, name_data: str, name_model: str, uniqid: str) -> None:
        path_model = generate_path_model_file(
            name_data, name_model, self.best_epoch, uniqid)
        ensure_file(path_model)
        # NOTE: avoid error when load gpu model from cpu environment
        th.save(self.model.to(th.device('cpu')).state_dict(), path_model)

    def predict(self, pretrain_path: str = '') -> UserItems:
        if pretrain_path != '':
            self.model = self.load_pretrained_model(self.model, pretrain_path)
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model.to(device)

        # ids
        users_batch_test = [th.LongTensor(self.dataset.uniq_users[i:i + 10000]).to(
            device) for i in range(0, self.dataset.n_users, 10000)]
        items = th.arange(self.dataset.n_items, dtype=th.long).to(device)

        time_start_eval = time()
        hits, precisions, recalls, ndcgs, dict_rec_items = self._evaluate_batch(
            dict_train_pos=self.dataset.dict_train_pos,
            dict_test_pos=self.dataset.dict_test_pos,
            users_batch=users_batch_test,
            items=items,
            list_k=self.top_ks,
            is_predict=True)

        logging.info('[Prediction] [{:.1f}s] : hits [{:s}], precision [{:s}], recall [{:s}], ndcg [{:s}]'.format(
            time() - time_start_eval,
            '\t'.join(['%.5f' % h for h in hits]),
            '\t'.join(['%.5f' % p for p in precisions]),
            '\t'.join(['%.5f' % r for r in recalls]),
            '\t'.join(['%.5f' % n for n in ndcgs]),
        ))
        return dict_rec_items
