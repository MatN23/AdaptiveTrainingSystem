#  Colossal-AI  ViT 

Yuxuan Lou, Mingyan Jiang

****
- [](../basics/booster_plugins.md)
- [booster API](../basics/booster_api.md)

****

- [Colossal-AI Examples ViT on `beans`](https://github.com/hpcaitech/ColossalAI/blob/main/examples/images/vit/vit_train_demo.py)

****
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)


## 

ViTColossal-AI  `beans`  ViT2-4 GPU


## 
1. Colossal-AI 
2. VIT
3.  [HybridParallelPlugin](../basics/booster_plugins.md) VIT
4. VIT

## Colossal-AI 
 Python  Colossal-AI 
```bash
pip install colossalai
```

## 

```python
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from data import BeansDataset, beans_collator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
```
##  Vision Transformer 

```python
SEED = 42
MODEL_PATH = "google/vit-base-patch16-224"
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
NUM_EPOCH = 3
WARMUP_RATIO = 0.3
TP_SIZE = 2
PP_SIZE = 2
```

```python
# Launch ColossalAI
colossalai.launch_from_torch(config={}, seed=SEED)
coordinator = DistCoordinator()
world_size = coordinator.world_size
```
criterionloss
, `BeansDataset`[data.py](https://github.com/hpcaitech/ColossalAI/blob/main/examples/images/vit/data.py)
```python
image_processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
train_dataset = BeansDataset(image_processor, TP_SIZE, split="train")
eval_dataset = BeansDataset(image_processor, RP_SIZE, split="validation")
num_labels = train_dataset.num_labels
```
VIT
```python
config = ViTConfig.from_pretrained(MODEL_PATH)
config.num_labels = num_labels
config.id2label = {str(i): c for i, c in enumerate(train_dataset.label_names)}
config.label2id = {c: str(i) for i, c in enumerate(train_dataset.label_names)}
model = ViTForImageClassification.from_pretrained(
    MODEL_PATH, config=config, ignore_mismatched_sizes=True
)
```
optimizer
```python
optimizer = HybridAdam(model.parameters(), lr=(LEARNING_RATE * world_size), weight_decay=WEIGHT_DECAY)
```
lr scheduler:
```python
total_steps = len(train_dataloader) * NUM_EPOCH
num_warmup_steps = int(WARMUP_RATIO * total_steps)
lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer, total_steps=(len(train_dataloader) * NUM_EPOCH), warmup_steps=num_warmup_steps
    )
```
criterion
```python
def _criterion(outputs, inputs):
    return outputs.loss
```
## VIT
colossalai`HybridParallelPlugin`[`HybridParallelPlugin`](../basics/booster_plugins.md)colossalai`HybridParallelPlugin`booster`booster.boost`
### 
`HybridParallelPlugin``precision`'fp16','bf16','fp32''fp16','bf16'`HybridParallelPlugin`zeroamp

amp
`initial_scale`AMP2**16
`min_scale`AMP1
`growth_factor`AMP2
`backoff_factor`AMP0.5
`growth_interval`AMP1000
`hysteresis`AMP2
`max_scale`AMP2**32

AMPplugin
```python
plugin = HybridParallelPlugin(
            precision="fp16",
            initial_scale=1,
        )
```

### 
`HybridParallelPlugin`shardformer`tp_size`

`enable_all_optimization`Shardformerflash attentionJITFalse
`enable_fused_normalization`ShardformerFalse
`enable_flash_attention`Shardformerflash attentionFalse
`enable_jit_fused`ShardformerJITFalse
`enable_sequence_parallelism`ShardformerFalse
`enable_sequence_overlap`ShardformerFalse

plugin
```python
plugin = HybridParallelPlugin(
            tp_size=4,
            enable_all_optimization=True
        )
```
### 
`HybridParallelPlugin``pp_size``num_microbatches`batchbatch`microbatch_size`batch`num_microbatches`micro batch
plugin
```python
plugin = HybridParallelPlugin(
            pp_size=4,
            num_microbatches=None,
            microbatch_size=1
        )
```
### 
`HybridParallelPlugin`zero-dptorch DDP`zero_stage`0()torch DDPtorch DDP`zero_stage`1zero1`zero_stage`2zero2,zero2zero3[`GeminiPlugin`](../basics/booster_plugins.md)zerozeroworld_size//(tp_size*pp_size)1`HybridParallelPlugin`torch DDP
torch DDP
`broadcast_buffers`DDPTrue
`ddp_bucket_cap_mb`DDPMB25
`find_unused_parameters`DDPFalse
`check_reductionDDPFalse
`gradient_as_bucket_view`DDPFalse
`static_graph`DDPFalse

Torch DDPplugin
```python
plugin = HybridParallelPlugin(
            tp_size=2,
            pp_size=1,
            zero_stage=0,
            precision="fp16",
            initial_scale=1,
        )
```
4torch DDP2.
zero
`zero_bucket_size_in_m`ZeRO12
`cpu_offload`ZeRO`cpu_offload`False
`communication_dtype`torchZeRONone
`overlap_communication`ZeROTrue

zero1plugin

```python
plugin = HybridParallelPlugin(
            tp_size=1,
            pp_size=1,
            zero_stage=1,
            cpu_offload=True,
            precision="fp16",
            initial_scale=1,
        )
```

### 
booster

```python
plugin = HybridParallelPlugin(
            tp_size=TP_SIZE,
            pp_size=PP_SIZE,
            num_microbatches=None,
            microbatch_size=1,
            enable_all_optimization=True,
            precision="fp16",
            initial_scale=1,
        )
booster = Booster(plugin=plugin)
```
`booster.boost`plugin
```python
model, optimizer, _criterion, train_dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, criterion=criterion, dataloader=train_dataloader, lr_scheduler=lr_scheduler
    )
```
##  ViT
`booster.execute_pipeline``scheduler`
```python
def run_forward_backward(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable[[Any, Any], torch.Tensor],
    data_iter: Iterator,
    booster: Booster,
):
    if optimizer is not None:
        optimizer.zero_grad()
    if isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1:
        # run pipeline forward backward when enabling pp in hybrid parallel plugin
        output_dict = booster.execute_pipeline(
            data_iter, model, criterion, optimizer, return_loss=True
        )
        loss, outputs = output_dict["loss"], output_dict["outputs"]
    else:
        batch = next(data_iter)
        batch = move_to_cuda(batch, torch.cuda.current_device())
        outputs = model(**batch)
        loss = criterion(outputs, None)
        if optimizer is not None:
            booster.backward(loss, optimizer)

def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: LRScheduler,
    dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    torch.cuda.synchronize()

    num_steps = len(dataloader)
    data_iter = iter(dataloader)
    enable_pbar = coordinator.is_master()
    if isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1:
        # when using pp, only the last stage of master pipeline (dp_rank and tp_rank are both zero) shows pbar
        tp_rank = dist.get_rank(booster.plugin.tp_group)
        dp_rank = dist.get_rank(booster.plugin.dp_group)
        enable_pbar = tp_rank == 0 and dp_rank == 0 and booster.plugin.stage_manager.is_last_stage()
    model.train()

    with tqdm(range(num_steps), desc=f"Epoch [{epoch + 1}]", disable=not enable_pbar) as pbar:
        for _ in pbar:
            loss, _ = run_forward_backward(model, optimizer, criterion, data_iter, booster)
            optimizer.step()
            lr_scheduler.step()

            # Print batch loss
            if enable_pbar:
                pbar.set_postfix({"loss": loss.item()})
```

```python
for epoch in range(NUM_EPOCH):
    train_epoch(epoch, model, optimizer, criterion, lr_scheduler, train_dataloader, booster, coordinator)
```
<!-- doc-test-command: echo  -->
