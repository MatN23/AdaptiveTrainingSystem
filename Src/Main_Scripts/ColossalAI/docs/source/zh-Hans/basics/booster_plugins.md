# Booster 

: [Hongxin Liu](https://github.com/ver217), [Baizhou Zhang](https://github.com/Fridge003), [Pengtai Xu](https://github.com/ppt0011)


**:**
- [Booster API](./booster_api.md)

## 

 [Booster API](./booster_api.md)  booster  booster 

:

- [Torch DDP ](#torch-ddp-):  `torch.nn.parallel.DistributedDataParallel` 
- [Torch FSDP ](#torch-fsdp-):  `torch.distributed.fsdp.FullyShardedDataParallel`  Zero-dp 
- [Low Level Zero ](#low-level-zero-):  `colossalai.zero.low_level.LowLevelZeroOptimizer` Zero-dp  Zero 12
- [Gemini ](#gemini-):  [Gemini](../features/zero_with_chunk.md)Gemini Chunk Zero-3
- [Hybrid Parallel ](#hybrid-parallel-): ShardformerTorchDDPZero-1/Zero-2transformerDDP, Zero



## 
- [Torch DDP ](#torch-ddp-):  20  Bert-3mGPT2-1.5b
- [Torch FSDP ](#torch-fsdp-) / [Low Level Zero ](#low-level-zero-):  100  GPTJ-6bMegatronLM-8b
- [Gemini ](#gemini-):  100  TuringNLG-17b**** Llama2-70b
- [Hybrid Parallel ](#hybrid-parallel-):  600 **** GPT3-175bBloom-176b

## 

### Low Level Zero 

 Zero-1  Zero-2/ CPU `reduce``gather`

Zero-1  Torch DDP 

Zero-2  Zero-2 

{{ autodoc:colossalai.booster.plugin.LowLevelZeroPlugin }}



- `timm.models.convit_base`
- dlrm and deepfm models in `torchrec`



### Gemini 

Chunk Zero-3 [Gemini ](../features/zero_with_chunk.md).

{{ autodoc:colossalai.booster.plugin.GeminiPlugin }}

### Hybrid Parallel 

Hybrid Parallel

1. Shardformer: Shardformer/ShardformerShardformerfused normalization, flash attention (xformers), JIT/Shardformer [Shardformer](../features/shardformer.md)ShardformerHybrid Parallel

<div align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/shardformer/shardformer_and_hybridparallel.png" width="500" />
</div>

2. fp16/bf16 [](../features/mixed_precision_training_with_booster.md)

3. Torch DDP: ZeroPytorch DDPTorch DDP [Pytorch DDP ](https://pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)

4. Zero: `zero_stage`12Zero 1/2Zero 1, Zero 2Zero [Low Level Zero ](#low-level-zero-).

>  , ShardformerHuggingface transformersLlama 1Llama 2OPTBloomBertGPT2transformersShardformer

{{ autodoc:colossalai.booster.plugin.HybridParallelPlugin }}

### Torch DDP 

 [Pytorch ](https://pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel).

{{ autodoc:colossalai.booster.plugin.TorchDDPPlugin }}

### Torch FSDP 

>   torch  1.12.0

>  / checkpoint

>  multi params groupoptimizer

 [Pytorch ](https://pytorch.org/docs/main/fsdp.html).

{{ autodoc:colossalai.booster.plugin.TorchFSDPPlugin }}

<!-- doc-test-command: echo  -->
