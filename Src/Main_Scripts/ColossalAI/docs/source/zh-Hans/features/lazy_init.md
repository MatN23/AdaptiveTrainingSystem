# 

: [Hongxin Liu](https://github.com/ver217)

**:**
- [Train with booster](../basics/booster_api.md)

## 



 `N`  `M` GB,  `4N >= M` 

## 

 booster 

### API 

{{ autodoc:colossalai.lazy.LazyInitContext }}

### 

```python
import colossalai
from colossalai.lazy import LazyInitContext
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin

from transformers import LlamaForCausalLM, LlamaConfig, BertForPreTraining

colossalai.launch({})
plugin = GeminiPlugin()
booster = Booster(plugin)

# 1. Initialize model from scratch
# Initialization on cuda will accelerate the initialization process but take more GPU memory.
with LazyInitContext(default_device="cuda"):
    model = LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=172, num_hidden_layers=4, num_attention_heads=4))
model, *_ = booster.boost(model)

# 2. Initialize model from pretrained
with LazyInitContext():
    model = BertForPreTraining.from_pretrained("prajjwal1/bert-tiny")
model, *_ = booster.boost(model)
```

>   colossalai>0.3.3 

## 

 booster 

|             |  |    |
|-----------------|---------|--------|
| Gemini          |        |        |
| Hybrid Parallel |        |        |
| Low Level Zero  |        |  |
| Torch DDP       |        |  |
| Torch FSDP      |        |  |

/

 torchvision, diffusers, timm, transformers, torchaudio  torchrec 

|                           |          |
|-------------------------------|--------------|
| wav2vec2_base                 | torchaudio   |
| hubert_base                   | torchaudio   |
| ViTModel                      | transformers |
| ViTForMaskedImageModeling     | transformers |
| ViTForImageClassification     | transformers |
| Blip2Model                    | transformers |
| Blip2ForConditionalGeneration | transformers |

<!-- doc-test-command: torchrun --standalone --nproc_per_node=2 lazy_init.py  -->
