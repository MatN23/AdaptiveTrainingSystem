#  GPT-2

: Hongxin Liu, Yongbin Li, Mingyan Jiang

****
- [](../basics/booster_plugins.md)
- [booster API](../basics/booster_api.md)

****
- [ColossalAI-Examples GPT2](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/gpt/hybridparallelism/finetune.py)

****
- [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

## 

 ViT--GPT-2GPT-2CPU

## 

:
1. 
2.  GPT-2 
3.  [HybridParallelPlugin](../basics/booster_plugins.md) GPT-2
4.  GPT-2

## 

```python
from typing import Callable, List, Union
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from tqdm import tqdm
from transformers import AutoConfig, GPT2ForSequenceClassification, get_linear_schedule_with_warmup
from transformers import AutoTokenizer

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
```
### plugin
[`HybridParallelPlugin`](../basics/booster_plugins.md)zero1.
```python
plugin = HybridParallelPlugin(
    tp_size=1,
    pp_size=2,
    num_microbatches=None,
    microbatch_size=1,
    enable_all_optimization=True,
    zero_stage=1,
    precision="fp16",
    initial_scale=1,
)
```

## .
```python
# Launch ColossalAI
colossalai.launch_from_torch(config={}, seed=42)
coordinator = DistCoordinator()
```
## GPT-2


```python
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
```
`plugin.prepare_dataloader`dataloader,dataloader
```python
def tokenize_batch(batch, tokenizer: Optional[AutoTokenizer] = None, max_length: int = 2048):
    texts = [sample["sentence1"] + sample["sentence2"] for sample in batch]
    data = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    data = {k: v.cuda() for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = datasets.load_dataset("glue", "mrpc")
train_dataloader = plugin.prepare_dataloader(
    dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    collate_fn=partial(tokenize_batch, tokenizer=tokenizer, max_length=512),
)
```
GPT-2
```python
cfg = AutoConfig.from_pretrained("gpt2", num_labels=2)
model = GPT2ForSequenceClassification.from_pretrained("gpt2", config=cfg).cuda()
```

```python
lr = LEARNING_RATE * coordinator.world_size
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, eps=1e-8)
```
 `lr_scheduler`  `criterion``criterion`loss
```python
# lr scheduler
total_steps = len(train_dataloader) * NUM_EPOCHS
num_warmup_steps = int(WARMUP_FRACTION * total_steps)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps,
)

def _criterion(outputs, inputs):
    return outputs.loss
```
## GPT-2
 HybridParallelPlugin  boosterboosterzero1
```python
booster = Booster(plugin=plugin)
```
 booster 
```python
model, optimizer, _criterion, _, lr_scheduler = booster.boost(
    model, optimizer, criterion=_criterion, lr_scheduler=lr_scheduler
)
```


##  GPT-2

 Booster  HybridParallelPlugin 
`booster.execute_pipeline`
```python
def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    _criterion: Callable,
    lr_scheduler: LRScheduler,
    train_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    print_flag = (not use_pipeline and coordinator.is_master()) or (use_pipeline and is_pp_last_stage)
    total_step = len(train_dataloader)

    model.train()
    optimizer.zero_grad()
    train_dataloader_iter = iter(train_dataloader)
    with tqdm(
        range(total_step),
        desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]",
        disable=not print_flag,
    ) as pbar:
        # Forward pass
        for _ in pbar:
            if use_pipeline:
                outputs = booster.execute_pipeline(
                    train_dataloader_iter, model, _criterion, optimizer, return_loss=True
                )
                # Backward and optimize
                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})
            else:
                data = next(train_dataloader_iter)
                data = move_to_cuda(data)
                outputs = model(**data)
                loss = _criterion(outputs, None)
                # Backward
                booster.backward(loss, optimizer)
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

```
 GPT-2 
```python
for epoch in range(NUM_EPOCHS):
    train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)
```
<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 train_gpt_using_hybrid_parallelism.py  -->
