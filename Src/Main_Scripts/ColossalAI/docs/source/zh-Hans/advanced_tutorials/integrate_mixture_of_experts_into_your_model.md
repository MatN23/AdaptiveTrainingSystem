#  MoE 

: Haichen Huang, Yongbin Li

****
- [ColossalAI-Examples WideNet](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/image/widenet)

****
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- [Go Wider Instead of Deeper](https://arxiv.org/abs/2107.11817)

## Introduction

`Switch Transformer` (MoE) 
Colossal-AI MoEColossal-AIMoEMoE

 MoE

## 

1. [MoE](#moe)
2. [MoE](#moe)
3. [](#)

[](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/image/widenet)  [ColossalAI-Examples](https://github.com/hpcaitech/ColossalAI-Examples).
 [WideNet](https://arxiv.org/abs/2107.11817)  MoE .

## MoE
`config.py` MoE`config.py``parallel``moe``moe`moe`moe`444GPU4moemo eGPUactivationGPU

```python
MOE_MODEL_PARALLEL_SIZE = ...
parallel = dict(
    moe=dict(size=MOE_MODEL_PARALLEL_SIZE)
)
```

`MOE_MODEL_PARALLEL_SIZE = E``E``E`transformer

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/oI59QcxdteKUTks.png"/>
<figcaption>MoE Transformer, image source: <a href="https://arxiv.org/abs/2006.16668">GShard</a></figcaption>
</figure>

GPUGPUmoe`WORLD_SIZE=4``MOE_MODEL_PARALLEL_SIZE=2`

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/Sn8FpmQPKIiBEq2.png"/>
<figcaption>MoE</figcaption>
</figure>

`MoeGradientHandler`all-reduce`colossalai.initialize`MoEMoE`colossalai.global_variables.moe_env`

```python
from colossalai.global_variables import moe_env
```

## MoE

`colossalai.nn.moe`MoE

```python
from colossalai.context.random import moe_set_seed
from model_zoo.moe.models import Widenet

moe_set_seed(42)
model = Widenet(num_experts=4, capacity_factor=1.2)
```

`moe_set_seed` moe

```python
from colossalai.nn.layer.moe import Experts, MoeLayer, Top2Router, NormalNoiseGenerator


noisy_func = NormalNoiseGenerator(num_experts)
shared_router = Top2Router(capacity_factor,
                           noisy_func=noisy_func)
shared_experts = Experts(expert=VanillaFFN,
                         num_experts=num_experts,
                         **moe_mlp_args(
                             d_model=d_model,
                             d_ff=d_ff,
                             drop_rate=drop_rate
                         ))
ffn=MoeLayer(dim_model=d_model, num_experts=num_experts,
             router=shared_router, experts=shared_experts)
```

ExpertsGPUexpert`Top1Router``Top2Router``colossalai.nn.layer.moe` expertsrouter`Moelayer``gate`API

## 

colossalai`colossalai.initialize` MoE `colossalai.initialize` `MoeGradientHandler`colossal`MoeGradientHandler`MoE`Moeloss`
```python
criterion = MoeLoss(
    aux_weight=0.01,
    loss_fn=nn.CrossEntropyLoss,
    label_smoothing=0.1
)
```
 `colossalai` `trainer``engine`

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 integrate_mixture_of_experts_into_your_model.py  -->
