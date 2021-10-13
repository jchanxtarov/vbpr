# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

import enum
from typing import Dict, List

# dataset
UserItems = Dict[int, List[int]]


class ModelType(enum.Enum):
    VBPR = "vbpr"
