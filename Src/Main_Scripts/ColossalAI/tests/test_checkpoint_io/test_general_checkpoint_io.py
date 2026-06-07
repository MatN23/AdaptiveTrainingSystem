import tempfile

import pytest
import torch
from torch.optim import Adam
from torchvision.models import resnet18

from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.testing import check_state_dict_equal, clear_cache_before_run, parameterize

# Note:
# 1. due to checkpoint IO can be quite slow if tested with all models, we will only test on resnet for now
# 2. we will test on both sharded and unsharded checkpoints
# 3. implement sharded checkpoint and test it


@clear_cache_before_run()
@parameterize("use_safetensors", [True, False])
def test_unsharded_checkpoint(use_safetensors: bool):
    model = resnet18()
    optimizer = Adam(model.parameters(), lr=0.001)
    lr_scheduler = CosineAnnealingWarmupLR(optimizer, total_steps=10)

    x = torch.randn(1, 3, 224, 224)

    # run fwd and bwd
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    if use_safetensors:
        suffix = ".safetensors"
    else:
        suffix = ".bin"
    model_ckpt_tempfile = tempfile.NamedTemporaryFile(suffix=suffix)
    optimizer_ckpt_tempfile = tempfile.NamedTemporaryFile()
    lr_scheduler_ckpt_tempfile = tempfile.NamedTemporaryFile()

    ckpt_io = GeneralCheckpointIO()
    ckpt_io.save_model(model, model_ckpt_tempfile.name, use_safetensors=use_safetensors)
    ckpt_io.save_optimizer(optimizer, optimizer_ckpt_tempfile.name)
    ckpt_io.save_lr_scheduler(lr_scheduler, lr_scheduler_ckpt_tempfile.name)

    new_model = resnet18()
    new_optimizer = Adam(new_model.parameters(), lr=0.001)
    new_lr_scheduler = CosineAnnealingWarmupLR(optimizer, total_steps=10)

    ckpt_io.load_model(new_model, model_ckpt_tempfile.name)
    ckpt_io.load_optimizer(new_optimizer, optimizer_ckpt_tempfile.name)
    ckpt_io.load_lr_scheduler(new_lr_scheduler, lr_scheduler_ckpt_tempfile.name)

    check_state_dict_equal(model.state_dict(), new_model.state_dict())
    check_state_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())


@pytest.mark.parametrize("use_safetensors", [True, False])
def test_sharded_model_checkpoint(use_safetensors: bool):
    model = resnet18()
    optimizer = Adam(model.parameters(), lr=0.001)
    x = torch.randn(1, 3, 224, 224)

    # run fwd and bwd
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    if use_safetensors:
        pass
    else:
        pass

    model_ckpt_dir = tempfile.TemporaryDirectory()
    optimizer_ckpt_tempfile = tempfile.NamedTemporaryFile()

    ckpt_io = GeneralCheckpointIO()

    ckpt_io.save_model(model, model_ckpt_dir.name, True, True, "", 10, use_safetensors=use_safetensors)
    ckpt_io.save_optimizer(optimizer, optimizer_ckpt_tempfile.name, shard=False)

    new_model = resnet18()
    new_optimizer = Adam(new_model.parameters(), lr=0.001)

    ckpt_io.load_model(new_model, str(model_ckpt_dir.name), strict=True)
    ckpt_io.load_optimizer(new_optimizer, optimizer_ckpt_tempfile.name)

    check_state_dict_equal(model.state_dict(), new_model.state_dict())
    check_state_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())


def test_sharded_optimizer_checkpoint():
    model = resnet18()
    optimizer = Adam(model.parameters(), lr=0.001)

    x = torch.randn(1, 3, 224, 224)

    # run fwd and bwd
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    model_ckpt_dir = tempfile.TemporaryDirectory()
    optimizer_ckpt_dir = tempfile.TemporaryDirectory()

    ckpt_io = GeneralCheckpointIO()

    ckpt_io.save_model(model, model_ckpt_dir.name, True, True, "", 10, use_safetensors=False)
    ckpt_io.save_optimizer(optimizer, optimizer_ckpt_dir.name, shard=True, size_per_shard=10)

    new_model = resnet18()
    new_optimizer = Adam(new_model.parameters(), lr=0.001)

    ckpt_io.load_model(new_model, str(model_ckpt_dir.name), strict=True)
    ckpt_io.load_optimizer(new_optimizer, str(optimizer_ckpt_dir.name))

    check_state_dict_equal(model.state_dict(), new_model.state_dict())
    check_state_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())

    # continue running fwd and bwd
    for _ in range(5):
        y = new_model(x)
        loss = y.sum()
        loss.backward()
        new_optimizer.step()

    ckpt_io.save_model(new_model, model_ckpt_dir.name, True, True, "", 10, use_safetensors=False)
    ckpt_io.save_optimizer(new_optimizer, optimizer_ckpt_dir.name, shard=True, size_per_shard=10)

    new_new_model = resnet18()
    new_new_optimizer = Adam(new_new_model.parameters(), lr=0.001)

    ckpt_io.load_model(new_new_model, str(model_ckpt_dir.name), strict=True)
    ckpt_io.load_optimizer(new_new_optimizer, str(optimizer_ckpt_dir.name))

    check_state_dict_equal(new_model.state_dict(), new_new_model.state_dict())
    check_state_dict_equal(new_optimizer.state_dict(), new_new_optimizer.state_dict())


def test_sharded_optimizer_multiple_param_groups():
    model = resnet18()
    optimizer = Adam(
        [{"params": model.layer1.parameters()}, {"params": model.layer2.parameters(), "lr": 0.002}], lr=0.001
    )

    x = torch.randn(1, 3, 224, 224)

    # run fwd and bwd
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    model_ckpt_dir = tempfile.TemporaryDirectory()
    optimizer_ckpt_dir = tempfile.TemporaryDirectory()

    ckpt_io = GeneralCheckpointIO()

    ckpt_io.save_model(model, model_ckpt_dir.name, True, True, "", 10, use_safetensors=False)
    ckpt_io.save_optimizer(optimizer, optimizer_ckpt_dir.name, shard=True, size_per_shard=10)

    new_model = resnet18()
    new_optimizer = Adam(
        [{"params": new_model.layer1.parameters()}, {"params": new_model.layer2.parameters(), "lr": 0.002}], lr=0.001
    )

    ckpt_io.load_model(new_model, str(model_ckpt_dir.name), strict=True)
    ckpt_io.load_optimizer(new_optimizer, str(optimizer_ckpt_dir.name))

    check_state_dict_equal(model.state_dict(), new_model.state_dict())
    check_state_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())