# Copyright (c) 2025 MatN23. All rights reserved.
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch.nn as nn
from torch.optim import Optimizer

from colossalai.interface import OptimizerWrapper


class MixedPrecision(ABC):
    """
    An abstract class for mixed precision training.
    """

    @abstractmethod
    def configure(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable]:
        # TODO: implement this method
        pass