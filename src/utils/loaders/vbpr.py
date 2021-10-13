# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

import logging

from .base import BaseDataset


class VbprDataset(BaseDataset):
    def __init__(self, dataset: str) -> None:
        super().__init__(dataset=dataset)

    def logging_statistics(self) -> None:
        logging.info("n_train: {}".format(self.n_train))
        logging.info("n_test:  {}".format(self.n_test))
        logging.info("n_users: {}".format(self.n_users))
        logging.info("n_items: {}".format(self.n_items))
