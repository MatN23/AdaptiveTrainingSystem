# 3D 

: Zhengda Bian, Yongbin Li

****
- [1D ](./1D_tensor_parallel.md)
- [2D ](./2D_tensor_parallel.md)

****
- [ColossalAI-Examples - 3D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)

****
- [Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/pdf/2105.14450.pdf)

## 

[3D ](https://arxiv.org/pdf/2105.14450.pdf) 

 $Y = XA$ 
 $P=q \times q \times q$ ,  $q=2$,  $X$  $A$ 

$$
\left[\begin{matrix}
            X_{000} & X_{001} \\
            X_{010} & X_{011} \\
            X_{100} & X_{101} \\
            X_{110} & X_{111} \end{matrix}
\right]
\text{~and~}
\left[\begin{matrix}
            A_{000} & A_{001} & A_{010} & A_{011} \\
            A_{100} & A_{101} & A_{110} & A_{111} \end{matrix}
\right]
\text{~respectively,}$$
 $X_{ijl}$  $A_{lji}$  $(i,j,l)$ , 

<center>
<img src="https://s2.loli.net/2022/02/17/JevO6SED5z4PFdp.png" width = "200" height = "250" />
<img src="https://s2.loli.net/2022/02/17/qvtwjdfNXMAb4nF.png" width = "200" height = "250" />
<img src="https://s2.loli.net/2022/02/17/WFzm2N4IwKf1jXZ.png" width = "200" height = "250" />
<img src="https://s2.loli.net/2022/02/17/r2dZQ4hKxwTuIv6.png" width = "200" height = "250" />
</center>

 $(i, 0...q,l)$  $X_{ijl}$, $(0...q, j, l)$  $A_{lji}$
 $(i,j,l)$  $X_{il}$  $A_{lj}$  $X_{il}A_{lj}$
 $(i, j, 0...q)$  reduce-scatter  $Y_{ijl}$, 
$$
Y=
\left[\begin{matrix}
            Y_{000} & Y_{001} \\
            Y_{010} & Y_{011} \\
            Y_{100} & Y_{101} \\
            Y_{110} & Y_{111} \end{matrix}
\right].
$$

,  all-gather  $\dot{Y_{ijl}}$,  reduce-scatter  $\dot{X_{il}}=\dot{Y_{ij}}A_{lj}^T$ and $\dot{A_{lj}}=X_{il}^T\dot{Y_{ij}}$

## 
 $P=q \times q \times q$ , 3D

|  |  () |  (activations) |  () |  () |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/q^3)$  | $O(1/q^3)$       | $O(1/q^3)$           | $O(6(q-1)/q^3)$           | $O(6(q-1))$             |

## 

ColossalAI3D3D`Shardformer``Shardformer`Shardformer

ColossalAI3D[ColossalAI-Examples - 3D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)

<!-- doc-test-command: echo  -->
