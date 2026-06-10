# Copyright (c) 2025 MatN23. All rights reserved.
from .cai_gptq import HAS_AUTO_GPTQ

if HAS_AUTO_GPTQ:
    from .cai_gptq import CaiGPTQLinearOp, CaiQuantLinear
    from .gptq_manager import GPTQManager