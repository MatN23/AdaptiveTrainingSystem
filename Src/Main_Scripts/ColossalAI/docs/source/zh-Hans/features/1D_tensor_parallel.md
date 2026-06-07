# 1D 

: Zhengda Bian, Yongbin Li


****
- [Tensor Parallelism with Shardformer](https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/shardformer/examples)

****
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf)

## 


[Megatron-LM](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf) 

 GEMM $Y = XA$ 2 $A$  $[A_1 ~ A_2]$,  $Y_i = XA_i$ ,  $[Y_1 ~ Y_2] = [XA_1 ~ XA_2]$. 

 $Z=YB$ ,  $B$ 
$$
\left[\begin{matrix} B_1 \\ B_2 \end{matrix} \right]
$$
.

$$
Z = [Y_1 ~ Y_2] \left[\begin{matrix} B_1 \\ B_2 \end{matrix} \right]
$$
 $Y_iB_i$ all-reduce $Z=Y_1B_1+Y_2B_2$

 $X$,  $i$  $\dot{X_i}=\dot{Y_i}A_i^T$all-reduce $\dot{X}=\dot{Y}A^T=\dot{Y_1}A_1^T+\dot{Y_2}A_2^T$

## 
 $P$ , 1D

|  |  () |  (activations) |  () |  () |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/P)$    | $O(1/P)$         | $O(1)$               | $O(2(P-1)/P)$             | $O(2(P-1))$             |


## 

ColossalAI1D`Shardformer`
`Shardformer`Shardformer

<!-- doc-test-command: echo  -->
