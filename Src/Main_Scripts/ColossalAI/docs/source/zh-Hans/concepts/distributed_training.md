# 

: Shenggui Li, Siqi Mai

## 

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/sE5daHf2ohIy9wX.png"/>
<figcaption>: <a href="https://towardsdatascience.com/distributed-training-in-the-cloud-cloud-machine-learning-engine-9e264ddde27f">Towards Data Science</a></figcaption>
</figure>



44


## 

2012[AlexNet](https://arxiv.org/abs/1404.5997) ImageNet GTX 580 3GB GPU GPU

1. 2015 [ResNet50](https://arxiv.org/abs/1512.03385) 2000
2018 [BERT-Large](https://arxiv.org/abs/1810.04805)3.452018
[GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
152020 [GPT-3](https://arxiv.org/abs/2005.14165) 17501000
<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/sCyreJ9PF1EdZYf.jpg"/>
<figcaption>: <a href="https://huggingface.co/blog/large-language-models">HuggingFace</a></figcaption>
</figure>


2. MNIST  CIFAR10  ImageNet JFT-300M 3 ImageNet-1k 300


3. GPU2012 K10 GPU 2020 A100 GPU

GPUGPUGPU


## 

/GPU

- host: (host)
- port: (port)
- rank: ID
- world size: 
- process group: (process group)

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/qnNBKh8AjzgM5sY.png"/>
<figcaption></figcaption>
</figure>

24 GPU84 GPU 

0295008 world size 8

| process ID | rank | Node index | GPU index |
| ---------- | ---- | ---------- | --------- |
| 0          | 0    | 0          | 0         |
| 1          | 1    | 0          | 1         |
| 2          | 2    | 0          | 2         |
| 3          | 3    | 0          | 3         |
| 4          | 4    | 1          | 0         |
| 5          | 5    | 1          | 1         |
| 6          | 6    | 1          | 2         |
| 7          | 7    | 1          | 3         |


:

| process ID | rank | Node index | GPU index |
| ---------- | ---- | ---------- | --------- |
| 0          | 0    | 0          | 0         |
| 2          | 1    | 0          | 2         |
| 4          | 2    | 1          | 0         |
| 6          | 3    | 1          | 2         |

**rank  rank rank  `world size of the process group - 1`**


1. peer-to-peer: 
2. collective: all-reduce

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/zTmlxgc3oeAdn97.png"/>
<figcaption>Collective communication : <a href="https://pytorch.org/tutorials/intermediate/dist_tuto.html">PyTorch distributed tutorial</a></figcaption>
</figure>
