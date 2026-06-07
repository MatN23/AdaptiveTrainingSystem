# Booster API

: [Mingyan Jiang](https://github.com/jiangmingyan), [Jianghai Chen](https://github.com/CjhHa1), [Baizhou Zhang](https://github.com/Fridge003)

**:**

- [](../concepts/distributed_training.md)
- [Colossal-AI ](../concepts/colossalai_overview.md)

****

<!-- update this url-->

- [BoosterCIFAR-10ResNet](https://github.com/hpcaitech/ColossalAI/blob/main/examples/tutorial/new_api/cifar_resnet)
- [BoosterRedPajamaLlama-1/2](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/llama2)

## 

 `colossalai.booster`  `colossalai.initialize` ()  booster API, .  `colossalai.booster` 
 `colossalai.booster` 

### Booster 

Booster eggemini  gemini 

**_HybridParallelPlugin:_** HybridParallelPlugin DDP, Zero

**_GeminiPlugin:_** GeminiPlugin  gemini  ZeRO 

**_TorchDDPPlugin:_** TorchDDPPlugin PytorchDDP

**_LowLevelZeroPlugin:_** LowLevelZeroPlugin  1/2  1 GPU  2 GPU 

**_TorchFSDPPlugin:_** TorchFSDPPlugin PytorchFSDPZeroDP

[Booster ](./booster_plugins.md)

[](../features/lazy_init.md)

### Booster 

<!--TODO: update autodoc -->

{{ autodoc:colossalai.booster.Booster }}

## 

 colossalai `booster.boost`  booster API 

 booster API :

```python
import torch
from torch.optim import SGD
from torchvision.models import resnet18

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin

def train():
    # launch colossalai
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')

    # create plugin and objects for training
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)
    model = resnet18()
    criterion = lambda x: x.mean()
    optimizer = SGD((model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # use booster.boost to wrap the training objects
    model, optimizer, criterion, _, scheduler = booster.boost(model, optimizer, criterion, lr_scheduler=scheduler)

    # do training as normal, except that the backward should be called by booster
    x = torch.randn(4, 3, 224, 224)
    x = x.to('cuda')
    output = model(x)
    loss = criterion(output)
    booster.backward(loss, optimizer)
    optimizer.clip_grad_by_norm(1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # checkpointing using booster api
    save_path = "./model"
    booster.save_model(model, save_path, shard=True, size_per_shard=10, use_safetensors=True)

    new_model = resnet18()
    booster.load_model(new_model, save_path)
```

Booster[](https://github.com/hpcaitech/ColossalAI/discussions/3046)

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 booster_api.py  -->
