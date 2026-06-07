# Colossal-AI5OPT

## 

[Colossal-AI](https://github.com/hpcaitech/ColossalAI)OPT

## Colossal-AI 
Colossal-AI  [Energon-AI](https://github.com/hpcaitech/EnergonAI) Colossal-AI

- **** Colossal-AIColossal-AI
- **** Colossal-AIOPT
- **** Colossal-AI(SIMD) 
- **** FastAPI left paddingbucket batching

## 

1. OPT

[](https://huggingface.co/patrickvonplaten/opt_metaseq_125m/blob/main/model/restored.pt)OPT-125M[](https://github.com/hpcaitech/EnergonAI/tree/main/examples/opt/script)

2. 

dockerhubColossal-AIdocker

```bash
docker pull hpcaitech/energon-ai:latest
```

3. HTTP

pythonHTTP [](https://github.com/hpcaitech/EnergonAI/tree/main/examples]) [OPT ](https://github.com/hpcaitech/EnergonAI/tree/main/examples/opt)
bash server.sh
 opt_config.py http
opt_125M

```bash
model_class = opt_125M
checkpoint = 'your_file_path'
```

gpu

```bash
tp_init_size = #gpu
```

docker`/model_checkpoint`  `/config`


```bash
export CHECKPOINT_DIR="your_opt_checkpoint_path"
# the ${CONFIG_DIR} must contain a server.sh file as the entry of service
export CONFIG_DIR="config_file_path"

docker run --gpus all  --rm -it -p 8020:8020 -v ${CHECKPOINT_DIR}:/model_checkpoint -v ${CONFIG_DIR}:/config --ipc=host energonai:latest
```

 `https://[IP-ADDRESS]:8020/docs#` 

## 

1. 

executor_max_batch_size  top_ktop_p 

```
executor_max_batch_size = 16
```

FIFO   executor_max_batch_size  opt-30b `executor_max_batch_size=16` opt-175b `executor_max_batch_size=4` 

2. 

config.py cache_size  cache_list_sizecache_list_size LRUcache_size=0

```
cache_size = 50
cache_list_size = 2
```
