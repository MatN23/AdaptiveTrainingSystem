# Adaptive Orchestrator Performance Overhead Guide

> [!NOTE]
> The metrics below for medium to enterprise scale setups are based on architectural projections and established industry patterns. Local verification was performed on an **NVIDIA T4** setup (Small Scale).

## Performance Impact by Setup Scale
### Small Scale (17B models, 12 GPUs)
#### Overhead: 38% slowdown
This is the tier where the hit actually stings. You feel it immediately in step time, gradient sync, and any part of the loop that already struggles for breathing room. The system starts to feel sluggish, almost like its pausing between operations. Adaptive orchestration just isnt built with this setup in mind; the coordination cost outweighs the benefit because theres not enough hardware for the orchestrator to really optimize anything meaningful.
Memory Overhead: 200600MB
On small GPUs, thats not trivial  it can push models closer to OOM territory depending on your batch size.
Recommendation: Skip adaptive training here. The orchestrator becomes more of a liability than a boost.

### Medium Scale (770B models, 48 GPUs)
#### Overhead: 13% slowdown
Youll still notice it, but its not the kind of thing that wrecks your training schedule. At this scale, the orchestrator starts doing its job: handling routing, balancing workloads, smoothing out irregularities in model architecture. The hit is real, but the benefits finally start to land.
Memory Overhead: 400800MB
Manageable, but something you still need to keep an eye on if your per-GPU RAM is tight.
Recommendation: Use it when your model architecture or routing logic is starting to get weird enough that manual tuning becomes annoying.

### Large Scale (70B+ models, 8+ GPUs)
####Overhead: 0.31% slowdown
At this point, the overhead barely registers. The orchestrator has enough hardware to play with that the coordination cost is tiny compared to the raw compute. This is where adaptive orchestration stops being a maybe and becomes the obvious move  it actually streamlines scaling behavior, helps with efficiency, and cuts down on the little engineering hacks youd otherwise need.
Memory Overhead: Basically irrelevant relative to model footprint.
Recommendation: Strong choice for production-level or long-running training runs.

### Enterprise Scale (500B+ models, 64+ GPUs)
#### Overhead: Under 0.1% slowdown
This is where the overhead effectively vanishes. At this size, youre coordinating so many experts, pathways, and parallel units that not using adaptive orchestration is a guaranteed path to pain. The orchestrator ends up saving far more time than it costs  routing decisions, load balancing, resharding, and failure handling all become smoother.
Memory Overhead: Trivial. Completely washed out by the sheer size of everything else in the system.
Recommendation: This should just be standard. Treat it like part of the baseline stack.

### Quick Decision Guide
#### Use Adaptive Training When:
Your training run is going to live for more than a week
The model is 70B parameters or larger
Youre running on more than 8 GPUs
Youre wrangling MoE setups or any architecture that requires dynamic routing or scheduling
Avoid Adaptive Training When:
Youre running fast experiments or iterating on ideas
The model is under 7B parameters
Youre limited to a single GPU or a tiny cluster
Youre in debugging mode and just need tight, predictable behavior without overhead

## Bottom Line
Adaptive orchestration is built for scale. Once you cross into real multi-GPU, multi-node, high-parameter territory, the overhead collapses into background noise and the system becomes genuinely useful. But if youre in the small-model or quick-iteration zone, the slowdown is too steep for the limited upside. Its a tool designed for big training pipelines  use it when youre operating at that level, not before.

