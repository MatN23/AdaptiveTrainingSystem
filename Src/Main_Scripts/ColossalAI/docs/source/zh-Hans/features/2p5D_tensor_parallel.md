# 2.5D 

: Zhengda Bian, Yongbin Li

****
- [1D ](./1D_tensor_parallel.md)
- [2D ](./2D_tensor_parallel.md)

****
- [ColossalAI-Examples - 2.5D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)

****
- [2.5-dimensional distributed model training](https://arxiv.org/pdf/2105.14500.pdf)

## 

[2.5D](https://arxiv.org/pdf/2105.14500.pdf)  2.5D SUMMA 

 $Y = XA$ 
 $P=q \times q \times d$ ,  $q=d=2$,  $X$  $d\times q$  $q$ 

$$
\left[\begin{matrix} X_{00} & X_{01} \\ X_{10} & X_{11} \\ X_{20} & X_{21} \\ X_{30} & X_{31}\end{matrix} \right],
$$
 $d$ 

$$
\left[\begin{matrix} X_{00} & X_{01} \\ X_{10} & X_{11} \end{matrix} \right] \text{~and~}\left[\begin{matrix} X_{20} & X_{21} \\ X_{30} & X_{31} \end{matrix} \right].
$$

 $A$ 

$$
\left[\begin{matrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{matrix} \right].
$$

 $X$ , SUMMA $X$  $A$ 


$$
\left[\begin{matrix} Y_{00}=X_{00}A_{00}+X_{01}A_{10} & Y_{01}=X_{00}A_{01}+X_{01}A_{11} \\ Y_{10}=X_{10}A_{00}+X_{11}A_{10} & Y_{11}=X_{10}A_{01}+X_{11}A_{11} \end{matrix} \right]
\text{~and~}
$$
$$
\left[\begin{matrix} Y_{20}=X_{20}A_{00}+X_{21}A_{10} & Y_{21}=X_{20}A_{01}+X_{21}A_{11} \\ Y_{30}=X_{30}A_{00}+X_{31}A_{10} & Y_{31}=X_{30}A_{01}+X_{31}A_{11} \end{matrix} \right].
$$

## 

 $P=q \times q \times d$ , 2.5D

|  |  () |  (activations) |  () |  () |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/dq^2)$ | $O(1/q^2)$       | $O(1/dq^2)$          | $\small O(3(q-1)(d+1)/dq)$       | $O(6(q-1))$             |

## 

ColossalAI2.5D2.5D`Shardformer``Shardformer`Shardformer

ColossalAI2.5D[ColossalAI-Examples - 2.5D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)

<!-- doc-test-command: echo  -->
