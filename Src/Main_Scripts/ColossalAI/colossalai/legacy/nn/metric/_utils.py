# Copyright (c) 2025 MatN23. All rights reserved.
import torch


def calc_acc(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    correct = torch.sum(targets == preds)
    return correct