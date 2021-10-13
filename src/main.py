# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

import logging
import random
from typing import Any

import numpy as np
import torch as th

from models.vbpr import VbprPredictor
from utils.helpers import (ensure_file, generate_path_log_file,
                           generate_path_pretrain_file,
                           generate_path_recitems_file,
                           save_recommended_items_list, set_logging)
from utils.loaders.vbpr import VbprDataset
from utils.parser import parse_args
from utils.types import ModelType


def initialize(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    path_logfile = generate_path_log_file(
        args.dataset, args.model, args.dt_now)
    if args.save_log:
        ensure_file(path_logfile)
    set_logging(path_logfile, args.save_log)


if __name__ == '__main__':
    args = parse_args()
    initialize(args)
    logging.info(args)
    dataset: Any
    model: Any

    if args.model == ModelType.VBPR.value:
        dataset = VbprDataset(args.dataset)
        dataset.logging_statistics()

        model = VbprPredictor(
            epochs=args.epochs,
            dim_embed_latent=args.dim_embed_latent,
            batch_size=args.batch_size,
            rates_reg=args.rates_reg,
            rate_learning=args.rate_learning,
            top_ks=args.top_ks,
            interval_evaluate=args.interval_evaluate,
            stopping_steps=args.stopping_steps,
        )

    model.load(dataset)

    if not args.use_pretrain:
        model.train()

        if args.save_model:
            model.save(args.dataset, args.model, args.dt_now)

    if args.predict:
        path_pretrain = ''
        if args.use_pretrain and args.pretrain_file:
            path_pretrain = generate_path_pretrain_file(
                args.dataset, args.pretrain_file)
        rec_items = model.predict(path_pretrain)

        if args.save_recommended_items:
            path_items = generate_path_recitems_file(
                args.dataset, args.model, args.dt_now)
            ensure_file(path_items)
            save_recommended_items_list(path_items, rec_items)
