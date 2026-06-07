#  Colossal-AI

: Chuanrui Wang, Shenggui Li, Siqi Mai

**:**
- [](../concepts/distributed_training.md)
- [Colossal-AI ](../concepts/colossalai_overview.md)


## 

 Colossal-AI  `launch` Colossal-AI

 Colossal-AI 
 `colossalai.launch`  `colossalai.get_default_parser`  SLURMOpenMPI  PyTorch  rank  world size 

 Colossal-AI 
-  colossalai.launch 
-  Colossal-AI 
-  SLURM 
-  OpenMPI 

## 

 Colossal-AI:
1. 
2. 

 Python dictionary 

### 

 `launch` , 
[](../concepts/distributed_training.md)  ``  :

1. host
2. port
3. rank
4. world_size
5. backend

 Colossal-AI  `colossalai.get_default_parser()`  `colossalai.launch` 

```python
# add these lines in your train.py
import colossalai

# get default parser
parser = colossalai.get_default_parser()

# if you want to add your own arguments
parser.add_argument(...)

# parse arguments
args = parser.parse_args()
```


```shell

python train.py --host <host> --rank <rank> --world_size <world_size> --port <port> --backend <backend>
```

`backend`  nccl

### 

 `colossalai.launch` API`colossalai.launch` 

```python
import colossalai

# parse arguments
args = colossalai.get_default_parser().parse_args()

# launch distributed environment
colossalai.launch(config=args.config,
                  rank=args.rank,
                  world_size=args.world_size,
                  host=args.host,
                  port=args.port,
                  backend=args.backend
)

```


###  Colossal-AI 

PyTorch
PyTorch

PyTorch`colossalai.launch_from_torch`
 rank, world size, host  port  PyTorch 

config.py
```python
BATCH_SIZE = 512
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 2
```
train.py
```python
import colossalai

colossalai.launch_from_torch(
    config="./config.py",
)
...
```

`colossalai run`4
`nproc_per_node`GPU`master_port`

```shell
# 4 29500
colossalai run --nproc_per_node 4 train.py

# 4
colossalai run --nproc_per_node 4 --master_port 29505 test.py
```

Colossal-AI

- `--hosts`

`host``host2`
`--master_addr` `master_addr``127.0.0.1`

:::caution

`master_addr``localhost``127.0.0.1`IP

:::

```shell
# 
colossalai run --nproc_per_node 4 --host host1,host2 --master_addr host1 test.py
```


- `--hostfile`

host file
SLURMPBS ProSLURM
`SLURM_NODELIST`PBS Pro`PBS_NODEFILE`
`echo $SLURM_NODELIST`  `cat $PBS_NODEFILE` 


Colossal-AIhost file

```text
host1
host2
```

host file`--host``master_addr`
host file
- `--include`: host file86
  `--include host1,host2,host3,...,host6`6

- `--exclude`: host1GPU
  `--exclude host1`

```shell
# hostfile
colossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  test.py

# 
colossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  --include host1 test.py

# 
colossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  --exclude host2 test.py
```


###  SLURM 

 SLURM   `srun`  Colossal-AI  `launch_from_slurm`  SLURM 
`launch_from_slurm`  `SLURM_PROCID`  `SLURM_NPROCS`  rank  world size 



```python
import colossalai

colossalai.launch_from_slurm(
    config=<CONFIG>,
    host=args.host,
    port=args.port
)
```



```bash
srun python train.py --host <master_node> --port 29500
```

###  OpenMPI 
OpenMPI `launch_from_openmpi` 
`launch_from_openmpi` 
`OMPI_COMM_WORLD_LOCAL_RANK` `MPI_COMM_WORLD_RANK`  `OMPI_COMM_WORLD_SIZE` local rankglobal rank  world size


```python
colossalai.launch_from_openmpi(
    config=<CONFIG>,
    host=args.host,
    port=args.port
)
```

 OpenMPI 
```bash
mpirun --hostfile <my_hostfile> -np <num_process> python train.py --host <node name or ip> --port 29500
```

- --hostfile: 
- --np: GPU --np 44 python  train.py
