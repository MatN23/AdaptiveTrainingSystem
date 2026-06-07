# Booster Checkpoint

: [Hongxin Liu](https://github.com/ver217)

**:**
- [Booster API](./booster_api.md)

## 

 [Booster API](./booster_api.md) booster  checkpoint

##  Checkpoint

{{ autodoc:colossalai.booster.Booster.save_model }}

 `colossalai.booster.Booster`  `checkpoint`  checkpoint   `shard=False` ,  `shard=True`checkpoint  checkpoint  checkpoint  [huggingface/transformers](https://github.com/huggingface/transformers) huggingface`from_pretrained`checkpoint

{{ autodoc:colossalai.booster.Booster.load_model }}

 `colossalai.booster.Booster`  checkpoint 

Huggingfacefrom_pretrained`booster.load``Lazy Initialization`
```python
from colossalai.lazy import LazyInitContext
from huggingface_hub import snapshot_download
...

# Initialize model under lazy init context
init_ctx = LazyInitContext(default_device=get_current_device)
with init_ctx:
     model = LlamaForCausalLM(config)

...

# Wrap the model through Booster.boost
model, optimizer, _, _, _ = booster.boost(model, optimizer)

# download huggingface pretrained model to local directory.
model_dir = snapshot_download(repo_id="lysandre/arxiv-nlp")

# load model using booster.load
booster.load(model, model_dir)
...
```

##  Checkpoint


{{ autodoc:colossalai.booster.Booster.save_optimizer }}

 `colossalai.booster.Booster` 

{{ autodoc:colossalai.booster.Booster.load_optimizer }}

 `colossalai.booster.Booster` 

##  Checkpoint

{{ autodoc:colossalai.booster.Booster.save_lr_scheduler }}

 `colossalai.booster.Booster`  `checkpoint`  checkpoint .

{{ autodoc:colossalai.booster.Booster.load_lr_scheduler }}

 `colossalai.booster.Booster`  `checkpoint`  checkpoint .

## Checkpoint 

 Checkpoint  [A Unified Checkpoint System Design](https://github.com/hpcaitech/ColossalAI/discussions/3339).

<!-- doc-test-command: echo  -->
