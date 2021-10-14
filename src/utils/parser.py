# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

import argparse
from datetime import datetime as dt
from typing import List

from utils.types import ModelType


def parse_args():
    parser = argparse.ArgumentParser()

    # about dataset
    parser.add_argument(
        '--seed',
        type=int,
        default=2020,
        help='Set random seed.',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='amazon-fashion',
        help='Select the target dataset.',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=ModelType.VBPR.value,
        choices=[model.value for model in ModelType],
        help='Select model.'
    )

    # about predictor
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Set max epochs.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Set batch size.'
    )
    parser.add_argument(
        '--rates_reg',
        type=List[float],
        default=[1e-4, 1e-4, 1e-4],
        help='Set reglarization rates.'
    )
    parser.add_argument(
        '--rate_learning',
        type=float,
        default=1e-2,
        help='Set rate_learning.'
    )
    parser.add_argument(
        '--top_ks',
        nargs='?',
        default='[20, 60, 100]',
        help='Set top_ks as list.'
    )
    parser.add_argument(
        '--interval_evaluate',
        type=int,
        default=3,
        help='Set interval_evaluate.'
    )
    parser.add_argument(
        '--stopping_steps',
        type=int,
        default=10,
        help='Set stopping_steps.'
    )
    parser.add_argument(
        '--dim_embed_latent',
        type=int,
        default=32,
        help='Set dim_embed_latent.'
    )
    parser.add_argument(
        '--dim_embed_visual',
        type=int,
        default=32,
        help='Set dim_embed_visual.'
    )

    # about experiment condition
    parser.add_argument(
        '--save_log',
        action='store_true',
        help='Whether saving log.'
    )
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Whether saving best model & attention.'
    )

    # for prediction
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Whether to predict.'
    )
    parser.add_argument(
        '--use_pretrain',
        action='store_true',
        help='Whether to use pretrained parapemters.'
    )
    parser.add_argument(
        '--save_recommended_items',
        action='store_true',
        help='Whether to save recommendation items list.'
    )

    args = parser.parse_args()
    args.dt_now = dt.now().strftime("%Y%m%d-%H%M%S")
    args.top_ks = eval(args.top_ks)

    args.is_sample_dataset = False
    if args.dataset == 'sample':
        args.is_sample_dataset = True

    return args
