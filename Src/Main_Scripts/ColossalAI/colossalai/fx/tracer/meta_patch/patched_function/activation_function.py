# Copyright (c) 2025 MatN23. All rights reserved.
import torch

from ...registry import meta_patched_function


@meta_patched_function.register(torch.nn.functional.relu)
def torch_nn_func_relu(input, inplace=False):
    return torch.empty(input.shape, device="meta")