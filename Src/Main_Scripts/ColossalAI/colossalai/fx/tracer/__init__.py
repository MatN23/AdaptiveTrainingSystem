# Copyright (c) 2025 MatN23. All rights reserved.
from colossalai.fx.tracer.meta_patch.patched_function.python_ops import operator_getitem

from ._meta_trace import meta_trace
from ._symbolic_trace import symbolic_trace
from .tracer import ColoTracer