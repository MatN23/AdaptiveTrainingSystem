# GeminiColossalAI

: [Jiarui Fang](https://github.com/feifeibear)

## 

GPU CPU  GPU  GPU CPU  CPU DRAM  NVMe SSDGPUGPUColossalAIGemini[PatrickStar](https://arxiv.org/abs/2108.05818)ColossalAI

## 

GeminiZeRObooster`GeminiPlugin``booster`[booster](../basics/booster_api.md)

```python
from torchvision.models import resnet18
from colossalai.booster import Booster
from colossalai.zero import ColoInitContext
from colossalai.booster.plugin import GeminiPlugin
plugin = GeminiPlugin(placement_policy='cuda', strict_ddp_mode=True, max_norm=1.0, initial_scale=2**5)
booster = Booster(plugin=plugin)
ctx = ColoInitContext()
with ctx:
    model = resnet18()
optimizer = HybridAdam(model.parameters(), lr=1e-3)
criterion = lambda x: x.mean()
model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)
)
```

GeminiTensor ParallelismData ParallelismPipeline ParallelismZeROTPPP

## 

****(**OP**erator)LinearLayerNorm



**(model data)**: 

**(non-model data)**:  GPU 

## 

DeepSpeed[Zero-offload](https://arxiv.org/abs/2101.06840)CPUGPU GPU CPUColossalAICPU

<figure style={{textAlign: "center"}}>
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/tutorial/gemini/deepspeed_compare.png"/>
<figcaption>Zero-OffloadGemini</figcaption>
</figure>


ColossalAIGeminiCPUGPUCPU-GPUGPUMemStatsCollector(MSC)StatefulTensorMgr(STM)


warmupnon-warmupwarmupMSCnon-warmupSTMMSCtensorCPU-GPUvolume

<figure style={{textAlign: "center"}}>
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/tutorial/gemini/gemini_workflow.png"/>
<figcaption>Gemini</figcaption>
</figure>


### StatefulTensorMgr

STMmodel data tensorColossalAImodel dataSTMHOLDCOMPUTEFREESTM

****tensorCPUGPU

****COMPUTEHOLDFREE

****COMPUTEHOLDTensor eviction strategyMSC


### MemStatsCollector
CPUGPUSTMMSCCPUGPU

**sampling moment)****period**bufferperiod-

preOpmodel data layout adjustperiodsystem memory usedperiodmodel data memory usedMSCZeROTensor ParallelOpgatherperiodMSCpreOpperiod 2-3tensor gathershard
gather bufferOpLinear OpTensor Parallelgather bufferOpZeROgather bufferPreOpPreOp


gather bufferOpLinear OpTensor Parallelgather bufferOpZeROgather bufferPreOpPreOp

<figure style={{textAlign: "center"}}>
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/tutorial/gemini/gemini_mem_curve.png"/>
<figcaption>Sampling based MemStatsCollector</figcaption>
</figure>

### Tensor Eviction Strategy

MSCtensor layoutS2model dataPeriod 2-3

warmup30%GPU

non-warmupPeriod
CPU-GPUtensor[cache thrashing](https://en.wikipedia.org/wiki/Thrashing_(computer_science))DNNOPT cachewarmuptensorHOLD tensortensor

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 meet_gemini.py  -->
