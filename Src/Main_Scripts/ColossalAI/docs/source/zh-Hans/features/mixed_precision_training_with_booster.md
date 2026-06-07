# 

: [Mingyan Jiang](https://github.com/jiangmingyan)

****

- [booster ](../basics/booster_api.md)

****

- [Accelerating Scientific Computations with Mixed Precision Algorithms](https://arxiv.org/abs/0808.2794)

## 

AMP 
 Colossal-AI , :

1. torch.cuda.amp
2. apex.amp
3. naive amp

| Colossal-AI    |  |  | fp16                                                  |
| -------------- | ------------ | ------------ | --------------------------------------------------------- |
| AMP_TYPE.TORCH |            |            |  fp16 |
| AMP_TYPE.APEX  |            |            |  opt_level O0, O1, O2, O3           |
| AMP_TYPE.NAIVE |            |            |  fp16             |

 PyTorch (1.6 )  NVIDIA Apex  Apex O2Apex-AMP  inf  nan torch amp 

>  fp16  ZeRO 
>
>   naive amp

 torch AMP NVIDIA AMP 

## 

:

1. [AMP ](#amp-)
2. [Colossal-AI  AMP](#colossal-ai--amp)
3. [](#)

## AMP 

 FP16  FP32 

FP16FP16  FP32  batch size 

 FP32 /

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/URzLJ3MPeDQbtck.png"/>
<figcaption>AMP  ( <a href="https://arxiv.org/abs/2108.05818">PatrickStar </a>)</figcaption>
</figure>

## Colossal-AI  AMP

 AMP  AMP booster  amp  booster `mixed_precision`;`bf16`,`pf8`.

#### booster 

 booster `mixed_precision="fp16"` torch amp

<!--- doc-test-ignore-start -->

```python
"""
    
    'fp16': torch amp
    'fp16_apex': apex amp,
    'bf16': bf16,
    'fp8': fp8,
    'fp16_naive': naive amp
"""
from colossalai import Booster
booster = Booster(mixed_precision='fp16',...)
```

<!--- doc-test-ignore-end -->

`FP16TorchMixedPrecision`

<!--- doc-test-ignore-start -->

```python
from colossalai.mixed_precision import FP16TorchMixedPrecision
mixed_precision = FP16TorchMixedPrecision(
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000)
booster = Booster(mixed_precision=mixed_precision,...)
```

<!--- doc-test-ignore-end -->

 amp 

### Torch AMP 

{{ autodoc:colossalai.booster.mixed_precision.FP16TorchMixedPrecision }}

### Apex AMP 

 Apex 
, O2  ( 2)  batch normalization  FP32

 [Apex Documentation](https://nvidia.github.io/apex/)

{{ autodoc:colossalai.booster.mixed_precision.FP16ApexMixedPrecision }}

### Naive AMP 

 Naive AMP ,  AMP  FP16  booster 

{{ autodoc:colossalai.booster.mixed_precision.FP16NaiveMixedPrecision }}

`colossalai.booster`,  AMP  GPU `dtype=torch.float16`

## 

 Colossal-AI  AMP Torch AMP.

###  1.  train.py 

`train.py`. `pip install timm scipy``scipy``timm`

```python
import os
from pathlib import Path

import torch
from timm.models import vit_base_patch16_224
from titans.utils import barrier_context
from torchvision import datasets, transforms

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR
```

###  2. 

`launch_from_torch` [Launch Colossal-AI](../basics/launch_colossalai.md)


```python
# 
parser = colossalai.get_default_parser()
args = parser.parse_args()

# launch from torch
colossalai.launch_from_torch(config=dict())

```

###  3. 

`DATA` `export DATA=/path/to/data`  `Path(os.environ['DATA'])`


```python
# define the constants
NUM_EPOCHS = 2
BATCH_SIZE = 128
# build model
model = vit_base_patch16_224(drop_rate=0.1)

# build dataloader
train_dataset = datasets.Caltech101(
    root=Path(os.environ['DATA']),
    download=True,
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Gray2RGB(),
        transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
    ]))

# build optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0.1)

# build loss
criterion = torch.nn.CrossEntropyLoss()

# lr_scheduler
lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=50, total_steps=NUM_EPOCHS)
```

###  4.  AMP

 MixedPrecision  torchDDPPlugin  `colossalai.boost`  FP16 .

```python
plugin = TorchDDPPlugin()
train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
booster = Booster(mixed_precision='fp16', plugin=plugin)

# if you need to customize the config, do like this
# >>> from colossalai.mixed_precision import FP16TorchMixedPrecision
# >>> mixed_precision = FP16TorchMixedPrecision(
# >>>     init_scale=2.**16,
# >>>     growth_factor=2.0,
# >>>     backoff_factor=0.5,
# >>>     growth_interval=2000)
# >>> plugin = TorchDDPPlugin()
# >>> booster = Booster(mixed_precision=mixed_precision, plugin=plugin)

# boost model, optimizer, criterion, dataloader, lr_scheduler
model, optimizer, criterion, dataloader, lr_scheduler = booster.boost(model, optimizer, criterion, dataloader, lr_scheduler)
```

###  5.  booster 

 booster 

```python
model.train()
for epoch in range(NUM_EPOCHS):
    for img, label in enumerate(train_dataloader):
        img = img.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        booster.backward(loss, optimizer)
        optimizer.step()
    lr_scheduler.step()
```

###  6. 

 `--nproc_per_node`  GPU

```shell
colossalai run --nproc_per_node 1 train.py
```

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 mixed_precision_training_with_booster.py  -->
