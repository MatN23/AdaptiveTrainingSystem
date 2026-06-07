# NVMe offload

: Hongxin Liu

**:**
- [Chunk (ZeRO)](../features/zero_with_chunk.md)

****

- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
## 

`N` Adam `8N` 32 GB  GPUGPU offload  GPU 

 Tensor I/O [TensorNVMe](https://github.com/hpcaitech/TensorNVMe) NVMe offload

> HDDSATA SSD  NVMe SSD HDD  SATA SSD  I/O  NVMe 

 offload I/O

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/08/16/CvRnowrsNyB4hza.jpg"/>
<figcaption></figcaption>
</figure>


## 

 [TensorNVMe](https://github.com/hpcaitech/TensorNVMe):

```shell
pip install packaging
pip install tensornvme
```

 Adam ([CPUAdam](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.nn.optimizer.cpu_adam.html)  [HybridAdam](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.nn.optimizer.hybrid_adam.html))  NVMe offload

<!--- doc-test-ignore-start -->

```python
from colossalai.nn.optimizer import CPUAdam, HybridAdam

optimizer = HybridAdam(model.parameters(), lr=1e-3, nvme_offload_fraction=1.0, nvme_offload_dir='./')
```

<!--- doc-test-ignore-end -->

`nvme_offload_fraction`  offload  NVMe  `nvme_offload_dir`  NVMe offload  `nvme_offload_dir`  `None`

 ColossalAI 


>   CPU  CPU  Zero/Gemini

## Examples

 --  GPT`transformers`



```shell
pip install psutil transformers
```



```python
import os
import time
from typing import Dict, Optional
import psutil
import torch
import torch.nn as nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
```



```python
class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
```



```python
def get_data(batch_size: int, seq_len: int,
             vocab_size: int, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
    device = torch.cuda.current_device() if device is None else device
    input_ids = torch.randint(vocab_size, (batch_size, seq_len),
                              device=device)
    attn_mask = torch.ones_like(input_ids)
    return dict(input_ids=input_ids, attention_mask=attn_mask)
def get_model_numel(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
def get_mem_usage() -> int:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss
```

 CPU  GPT 

```python
def train_cpu(nvme_offload_fraction: float = 0.0):
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    criterion = GPTLMLoss()
    optimizer = HybridAdam(model.parameters(), nvme_offload_fraction=nvme_offload_fraction)
    print(f'Model numel: {get_model_numel(model) / 1024**3:.3f} B')
    start = time.time()
    for step in range(3):
        data = get_data(4, 128, config.vocab_size, device='cpu')
        outputs = model(**data)
        loss = criterion(outputs.logits, data['input_ids'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'[{step}] loss: {loss.item():.3f}')
    print(f'Time: {time.time() - start:.3f} s')
    print(f'Mem usage: {get_mem_usage() / 1024**2:.3f} MB')
```

 NVME 

```python
train_cpu(0.0)
```



```
Model numel: 0.116 B
[0] loss: 10.953
[1] loss: 10.974
[2] loss: 10.965
Time: 7.739 s
Mem usage: 5966.445 MB
```

 NVME 

```python
train_cpu(1.0)
```



```
Model numel: 0.116 B
[0] loss: 10.951
[1] loss: 10.994
[2] loss: 10.984
Time: 8.527 s
Mem usage: 4968.016 MB
```

1.16 GPT2-S  0.928 GB NVME  998 MB 

 Gemini  GPT `"auto"` `"cpu"`  `"const"`

```python
def train_gemini_cpu(nvme_offload_fraction: float = 0.0):
    colossalai.launch_from_torch({})
    config = GPT2Config()
    with ColoInitContext(device=torch.cuda.current_device()):
        model = GPT2LMHeadModel(config)
    criterion = GPTLMLoss()
    optimizer = HybridAdam(model.parameters(), nvme_offload_fraction=nvme_offload_fraction)
    print(f'Model numel: {get_model_numel(model) / 1024**3:.3f} B')

    plugin = GeminiPlugin(
                strict_ddp_mode=True,
                device=torch.cuda.current_device(),
                placement_policy='cpu',
                pin_memory=True,
                hidden_dim=config.n_embd,
                initial_scale=2**5
                )
    booster = Booster(plugin)
    model, optimizer, criterion, _* = booster.boost(model, optimizer, criterion)

    start = time.time()
    for step in range(3):
        data = get_data(4, 128, config.vocab_size)
        outputs = model(**data)
        loss = criterion(outputs.logits, data['input_ids'])
        booster.backward(loss, optimizer)
        optimizer.step()
        optimizer.zero_grad()
        print(f'[{step}] loss: {loss.item():.3f}')
    print(f'Time: {time.time() - start:.3f} s')
    print(f'Mem usage: {get_mem_usage() / 1024**2:.3f} MB')
```

 NVME 

```python
train_gemini_cpu(0.0)
```



```
Model numel: 0.116 B
searching chunk configuration is completed in 0.27 s.
used number: 118.68 MB, wasted number: 0.75 MB
total wasted percentage is 0.63%
[0] loss: 10.953
[1] loss: 10.938
[2] loss: 10.969
Time: 2.997 s
Mem usage: 5592.227 MB
```

 NVME 

```python
train_gemini_cpu(1.0)
```



```
Model numel: 0.116 B
searching chunk configuration is completed in 0.27 s.
used number: 118.68 MB, wasted number: 0.75 MB
total wasted percentage is 0.63%
[0] loss: 10.953
[1] loss: 10.938
[2] loss: 10.969
Time: 3.691 s
Mem usage: 5298.344 MB
```

NVME  294 MB  Gemini  `pin_memory`  `pin_memory` 900 MB 

## API 

{{ autodoc:colossalai.nn.optimizer.HybridAdam }}

{{ autodoc:colossalai.nn.optimizer.CPUAdam }}


<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 nvme_offload.py  -->
