# 

: Guangyang Lu, Hongxin Liu, Yongbin Li, Mingyan Jiang

****
- [](../concepts/paradigms_of_parallelism.md)
- [Booster API](../basics/booster_api.md)
- [Shardformer](../features/shardformer.md)
- [Booster ](../basics/booster_plugins.md)

****
- [pipelineBert](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/bert/finetune.py)

****
- [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)

## 

 Colossal-AI ,  NVIDIA  1F1B ,  ViT  ImageNet  Bert  Glue .

## 

:

1.  1F1B 
2.  schedule
3.  Bert

##  1F1B 

 GPipe

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/OAucPF6mWYynUtV.png"/>
<figcaption>1: GPipe <a href="https://arxiv.org/pdf/2104.04473.pdf">Megatron-LM</a> </figcaption>
</figure>

 GPipe microbatches 

1F1B GPipe 1F1B  schedule 
<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/iJrVkp2HLcahjsT.png"/>
<figcaption>Figure2:  <a href="https://arxiv.org/pdf/2104.04473.pdf">Megatron-LM</a>  schedule schedule</figcaption>
</figure>

###  Schedule

 schedule 

 GPipe  GPipe 

###  Schedule

 schedule **microbatches**

 schedule 11-425-811,2,9,1023,4,11,12




## Colossal-AI

 Colossal-AI  `scheduler`  `Shardformer``OneForwardOneBackwardSchedule``InterleavedSchedule` Shardformer  `forward` 

 Colossal-AI `HybridParallelPlugin`  `scheduler` `shardformer.optimize`  `execute_pipeline`  `scheduler`  `HybridParallelPlugin``OneForwardOneBackwardSchedule`, `InterleavedSchedule`

 `HybridParallelPlugin` `HybridParallelPlugin`[](../basics/booster_plugins.md)

##  Bert

`model`, `dataloader`, `optimizer`, `lr_scheduler`, `criterion` :
```python
import argparse
from typing import Callable, List, Union

import torch
import torch.nn as nn
from data import GLUEDataBuilder
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AlbertForSequenceClassification,
    AutoConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

# Define some config
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1

coordinator = DistCoordinator()

def move_to_cuda(batch):
    return {k: v.cuda() for k, v in batch.items()}

# Define 'criterion' function with two inputs, which will be passed to 'execute_pipeline'.
def _criterion(outputs, inputs):
    return outputs.loss

# Define optimizer
lr = LEARNING_RATE
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


# Define lr_scheduler
total_steps = len(train_dataloader) * NUM_EPOCHS
num_warmup_steps = int(WARMUP_FRACTION * total_steps)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps,
)


# Define Bert model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=cfg).cuda()

# Define a dataloader
data_builder = GLUEDataBuilder(model_name,
                                plugin,
                                args.task,
                                train_batch_size=BATCH_SIZE,
                                eval_batch_size=BATCH_SIZE)
train_dataloader = data_builder.train_dataloader()
```

`HybridParallelPlugin`booster.
```python
plugin = HybridParallelPlugin(tp_size=1,
                                pp_size=2,
                                num_microbatches=None,
                                microbatch_size=1,
                                enable_all_optimization=True,
                                zero_stage=1,
                                precision='fp16',
                                initial_scale=1)
booster = Booster(plugin=plugin)
```

`booster`
```python
model, optimizer, _criterion, _, lr_scheduler = booster.boost(model,
                                                                optimizer,
                                                                criterion=_criterion,
                                                                lr_scheduler=lr_scheduler)
```


```python
# Define a train function
def train_epoch(epoch: int, model: nn.Module, optimizer: Optimizer, _criterion: Callable, lr_scheduler: LRScheduler,
                train_dataloader: DataLoader, booster: Booster, coordinator: DistCoordinator):

    is_pp_last_stage = booster.plugin.stage_manager.is_last_stage()
    total_step = len(train_dataloader)

    model.train()
    optimizer.zero_grad()
    # convert train_dataloader to a iterator
    train_dataloader_iter = iter(train_dataloader)
    with tqdm(range(total_step),
              desc=f'Epoch [{epoch + 1}/{NUM_EPOCHS}]',
              disable=not (is_pp_last_stage)) as pbar:
        # Forward pass
        for _ in pbar:
            outputs = booster.execute_pipeline(train_dataloader_iter,
                                                model,
                                                _criterion,
                                                optimizer,
                                                return_loss=True)
            # Backward and optimize
            if is_pp_last_stage:
                loss = outputs['loss']
                pbar.set_postfix({'loss': loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

# Train model
for epoch in range(NUM_EPOCHS):
    train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)
```

 `2`  batch  `1`  micro batches
<!-- doc-test-command: echo  -->
