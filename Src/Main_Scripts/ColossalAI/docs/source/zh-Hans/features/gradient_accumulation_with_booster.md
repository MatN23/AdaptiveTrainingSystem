# 

: [Mingyan Jiang](https://github.com/jiangmingyan), [Baizhou Zhang](https://github.com/Fridge003)

****
- [Booster](../basics/booster_api.md)

## 

 batch size   batch size 2

## 

 Colossal-AI boosterno_sync

## 

4

###  1.  train.py 
train.py `torch` 1.8.1

```python
import os
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.logging import get_dist_logger
from colossalai.cluster.dist_coordinator import priority_execution
```

###  2. 

`launch_from_torch` [Launch Colossal-AI](../basics/launch_colossalai.md)

```python
# initialize distributed setting
parser = colossalai.get_default_parser()
args = parser.parse_args()

# launch from torch
colossalai.launch_from_torch(config=dict())

```

###  3. 

`DATA` `export DATA=/path/to/data`  `Path(os.environ['DATA'])`

```python
# define the training hyperparameters
BATCH_SIZE = 128
GRADIENT_ACCUMULATION = 4

# build resnet
model = resnet18(num_classes=10)

# build dataloaders
with priority_execution():
    train_dataset = CIFAR10(root=Path(os.environ.get('DATA', './data')),
                            download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(size=32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                            ]))

# build criterion
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
```

###  4. 
`TorchDDPPlugin``Booster` `booster.boost`

```python
plugin = TorchDDPPlugin()
booster = Booster(plugin=plugin)
train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
model, optimizer, criterion, train_dataloader, _ = booster.boost(model=model,
                                                                    optimizer=optimizer,
                                                                    criterion=criterion,
                                                                    dataloader=train_dataloader)
```


###  5. booster
booster `param_by_iter` 
```python
optimizer.zero_grad()
for idx, (img, label) in enumerate(train_dataloader):
        sync_context = booster.no_sync(model)
        img = img.cuda()
        label = label.cuda()
        if idx % (GRADIENT_ACCUMULATION - 1) != 0:
            with sync_context:
                output = model(img)
                train_loss = criterion(output, label)
                train_loss = train_loss / GRADIENT_ACCUMULATION
                booster.backward(train_loss, optimizer)
        else:
            output = model(img)
            train_loss = criterion(output, label)
            train_loss = train_loss / GRADIENT_ACCUMULATION
            booster.backward(train_loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()

        ele_1st = next(model.parameters()).flatten()[0]
        param_by_iter.append(str(ele_1st.item()))

        if idx != 0 and idx % (GRADIENT_ACCUMULATION - 1) == 0:
            break

    for iteration, val in enumerate(param_by_iter):
        print(f'iteration {iteration} - value: {val}')

    if param_by_iter[-1] != param_by_iter[0]:
        print('The parameter is only updated in the last iteration')

```

###  6. 

```shell
colossalai run --nproc_per_node 1 train.py
```

3

```text
iteration 0, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 1, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 2, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 3, first 10 elements of param: tensor([-0.0141,  0.0464,  0.0507,  0.0321,  0.0356, -0.0150,  0.0172, -0.0118, 0.0222,  0.0473], device='cuda:0', grad_fn=<SliceBackward0>)
```

## Gemini

`no_sync()` `TorchDDPPlugin`  `LowLevelZeroPlugin``stage`1. `GeminiPlugin`  `no_sync()` , `pytorch`

`GeminiPlugin``enable_gradient_accumulation``True` `GeminiPlugin` :
<!--- doc-test-ignore-start -->
```python
...
plugin = GeminiPlugin(..., enable_gradient_accumulation=True)
booster = Booster(plugin=plugin)
...

...
for idx, (input, label) in enumerate(train_dataloader):
    output = gemini_model(input.cuda())
    train_loss = criterion(output, label.cuda())
    train_loss = train_loss / GRADIENT_ACCUMULATION
    booster.backward(train_loss, gemini_optimizer)

    if idx % (GRADIENT_ACCUMULATION - 1) == 0:
        gemini_optimizer.step() # zero_grad is automatically done
...
```
<!--- doc-test-ignore-end -->

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 gradient_accumulation_with_booster.py  -->
