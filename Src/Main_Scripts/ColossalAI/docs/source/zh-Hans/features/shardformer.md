# Shardformer

Author: [Baizhou Zhang](https://github.com/Fridge003), [Bin Jia](https://github.com/FoolPlayer)

****
- [](../concepts/paradigms_of_parallelism.md)
- [Booster API](../basics/booster_api.md)
- [Booster ](../basics/booster_plugins.md)

****
- [Shardformer](https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/shardformer/examples)
- [HybridParallelPluginShardformer](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/bert)

****
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)


## 

LLaMa-2 70BOPT 175BTransformerGPU/Huggingface transformersHuggingface transformers

ColossalAI**Shardformer**HuggingFaceTransformertransformersShardformer/

## 

/ 

<table>
  <tr>
    <th nowrap="nowrap">Model/Feature</th>
    <th nowrap="nowrap" title="Tensor Parallel">Tensor<br />Parallel</th>
    <th nowrap="nowrap" align="center" title="Pipeline Parallel">Pipeline<br />Parallel</th>
    <th nowrap="nowrap" align="center" title="Lazy Initialization">Lazy<br />Initialization</th>
    <th nowrap="nowrap" align="center" title="xFormers">xFormers</th>
    <th nowrap="nowrap" align="center" title="Flash Attention 2">Flash<br />Attention 2</th>
    <th nowrap="nowrap" align="center" title="JIT Fused Operators">JIT Fused<br />Operators</th>
    <th nowrap="nowrap" align="center" title="Fused LayerNorm">Fused<br />LayerNorm</th>
    <th nowrap="nowrap" align="center" title="Sequence Parallel">Sequence<br />Parallel</th>
    <th nowrap="nowrap" align="center" title="Sequence Overlap">Sequence<br />Overlap</th>
  </tr>
  <tr>
    <td nowrap="nowrap">Llama V1/V2</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
  <tr>
    <td nowrap="nowrap">OPT</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
    <tr>
    <td nowrap="nowrap">BLOOM</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
  <tr>
    <td nowrap="nowrap">ChatGLM 2</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
  <tr>
    <td nowrap="nowrap">BERT</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
  <tr>
    <td nowrap="nowrap">GPT 2</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
  <tr>
    <td nowrap="nowrap">T5</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
  <tr>
    <td nowrap="nowrap">ViT</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
  <tr>
    <td nowrap="nowrap">Whisper</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
  <tr>
    <td nowrap="nowrap">SAM</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
  <tr>
    <td nowrap="nowrap">Blip2</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
  <tr>
    <td nowrap="nowrap">Falcon</td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
    <td nowrap="nowrap" align="center"></td>
  </tr>
  <tr>
    <td colspan="39"></td>
  </tr>
</table>

Shardformer:
- RoBERTa
- ALBERT
- ERNIE
- GPT Neo
- GPT-J
- BEiT
- SwinTransformer V1/V2
- qwen

//[Issues](https://github.com/hpcaitech/ColossalAI/issues)

## 

### Shardformer

Shardformer`ShardConfig`

{{ autodoc:colossalai.shardformer.ShardConfig }}

 Apex Fused Layernorm `apex` flash attention `flash_attn`xFormers  `cutlass_op` Flash Attention

### Shardformer

#### 1. BoosterShardformer ()

`HybridParallelPlugin``Booster``Shardformer``Booster``execute_pipeline``HybridParallelPlugin``Shardformer`Zero

[](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/bert)`HybridParallelPlugin``Shardformer`

```bash
torchrun --standalone --nproc_per_node 4  finetune.py --target_f1 0.86 --plugin "hybrid_parallel" --model_type "bert"
```
`Shardformer`Bert`HybridParallelPlugin`

`finetune.py`

`main`
```python
...
elif args.plugin == "hybrid_parallel":
    # modify the param accordingly for finetuning test cases
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
`tp_size`, `pp_size`  `zero_stage`[Booster ](../basics/booster_plugins.md)

 Booster

1. criterionloss:
    ```python
    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss
    ```

2.  `train_epoch` , dataloader  `Iterator` :
    ```python
    train_dataloader_iter = iter(train_dataloader)
    ```

3. `Booster.execute_pipeline` :
    ```python
    outputs = booster.execute_pipeline(
        train_dataloader_iter, model, _criterion, optimizer, return_loss=True
    )
    ```
     `loss.backward()`
     `Booster.execute_pipeline`  [Booster API ](../basics/booster_api.md)

#### 2. Shardformer APIShardformer ()

Shardformer APIShardformer`Booster`

[](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/examples/convergence_benchmark.py)
ShardformerAPI`Shardformer`
`train`
```python
...
if dist.get_world_size() > 1:
    tp_group = dist.new_group(backend="nccl")

    # First create configuration for Shardformer
    shard_config = ShardConfig(
        tensor_parallel_process_group=tp_group,
        enable_tensor_parallelism=True,
        enable_all_optimization=True
    )

    # Then create ShardFormer object with created config
    shard_former = ShardFormer(shard_config=shard_config)

    # Finally shard the model using ShardFormer.optimize method
    model, _ = shard_former.optimize(model)
...
```

### 

1. `model(input)``loss.backward()`/`booster.execute_pipeline`/

2. Shardformer`GPT2ForSequenceClassification``ViTForImageClassification`labelsShardformerclassifiertransformersconfigbug Shardformer


## Shardformer

### 

Shardformer

1. PyTorch`nn.Linear``nn.Embedding`
split/gather`from_native_module`PyTorch

2. Huggingface Transformers2LlaMa-2,`LlamaDecoderLayer`   `num_heads``model.config.num_attention_heads // 2`

3. Huggingface transformers`flash attention`

4. `ModelSharder.shard`


ShardformerShardformer[Shardformer ](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/README.md)[](https://github.com/hpcaitech/ColossalAI/discussions/4050)

###  Sequence Parallelism

`Shardformer``Shardformer`[](https://colossalai.org/docs/basics/configure_parallelization/#sequence-parallel)ring attention`Shardformer`1Dactivation

1. [1D](https://colossalai.org/docs/features/1D_tensor_parallel)$g$$\vec{g}$$g$$\vec{g}$All-Reduce

2. $\vec{g}$All-GatherReduce-Scatter$\vec{g}$Reduce-ScatterAll-Gather

3. NCCLAll-reduce`Ring All-Reduce`Reduce-ScatterAll-Gather

4.  `Column Linear` $(batch, sequence\_len/k, hidden\_states)$`Shardformer``enable_sequence_overlap`


<!-- doc-test-command: echo  -->
