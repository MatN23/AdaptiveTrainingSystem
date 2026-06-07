# 2D 

: Zhengda Bian, Yongbin Li

****
- [1D ](./1D_tensor_parallel.md)

****
- [ColossalAI-Examples - 2D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)

****
- [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/pdf/2104.05343.pdf)

## 

1D activations 
 SUMMA [2D](https://arxiv.org/pdf/2104.05343.pdf) 

 $Y = XA$ 
 $P=q\times q$ ,  $q=2$,  $X$ A $A$ 

$$
\left[\begin{matrix} X_{00} & X_{01} \\ X_{10} & X_{11} \end{matrix} \right]
\text{~and~}
\left[\begin{matrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{matrix} \right].
$$

 $q$   $t=1$ , $X_{i0}$ ,  $A_{0j}$ 

$$
\left[\begin{matrix} X_{00},A_{00} & X_{00},A_{01} \\ X_{10},A_{00} & X_{10},A_{01} \end{matrix} \right].
$$

 $(i, j)$  $X_{i0}$  $A_{0j}$ 

$$
\left[\begin{matrix} X_{00}A_{00} & X_{00}A_{01} \\ X_{10}A_{00} & X_{10}A_{01} \end{matrix} \right] (1).
$$

 $t=2$ , $X_{i1}$ , $A_{1j}$ , 

$$
\left[\begin{matrix} X_{01}A_{10} & X_{01}A_{11} \\ X_{11}A_{10} & X_{11}A_{11} \end{matrix} \right] (2).
$$

 $(1)$  $(2)$ 

$$
Y = XA = \left[\begin{matrix} X_{00}A_{00}+X_{01}A_{10} & X_{00}A_{01}+X_{01}A_{11} \\ X_{10}A_{00}+X_{11}A_{10} & X_{10}A_{01}+X_{11}A_{11} \end{matrix} \right].
$$

## 
 $P=q\times q$ , 2D

|  |  () |  (activations) |  () |  () |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/q^2)$  | $O(1/q^2)$       | $O(1/q^2)$           | $O(6(q-1)/q)$             | $O(6(q-1))$             |

## 

ColossalAI2D2D`Shardformer``Shardformer`Shardformer

ColossalAI2D[ColossalAI-Examples - 2D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)

<!-- doc-test-command: echo  -->
