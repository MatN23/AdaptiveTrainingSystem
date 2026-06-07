# 

: Shenggui Li, Siqi Mai

## 

 GPU [GitHub](https://github.com/hpcaitech/ColossalAI/discussions)

## 



<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/WSAensMqjwHdOlR.png"/>
<figcaption></figcaption>
</figure>

## 

 GPU -

### 

 `N`  `1/N`

 `C = AB`B `[B0 B1 B2 ... Bn]` `A`  `B`  `[AB0 AB1 AB2 ... ABn]` (rank=0) `AB0`

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/2ZwyPDvXANW4tMG.png"/>
<figcaption></figcaption>
</figure>

 Colossal-AI  1D2D2.5D  3D ``


:
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/abs/2104.05343)
- [2.5-dimensional distributed model training](https://arxiv.org/abs/2105.14500)
- [Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/abs/2105.14450)

### 

 CPU 

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/at3eDv7kKBusxbd.png"/>
<figcaption></figcaption>
</figure>



<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/sDNq51PS3Gxbw7F.png"/>
<figcaption>Source: <a href="https://arxiv.org/abs/1811.06965">GPipe</a></figcaption>
</figure>

:
- [PipeDream: Fast and Efficient Pipeline Parallel DNN Training](https://arxiv.org/abs/1806.03377)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines](https://arxiv.org/abs/2107.06925)


## 

 `ZeRO`[](https://arxiv.org/abs/1910.02054) ZeRO ZeROfp16

- Level 1: 
- Level 2: 32
- Level 3: 16

:
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)


## 

 GPU  GPU CPU CPU GB GPU 1632GB CPU 

 CPU  NVMe  CPU  NVMe 

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/qLHD5lk97hXQdbv.png"/>
<figcaption></figcaption>
</figure>

:
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management](https://arxiv.org/abs/2108.05818)
