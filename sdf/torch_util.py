from typing import Any, Union

import numpy as np
import torch
torch.set_default_dtype(torch.float64)


device = "cpu"
REF_TENSOR = torch.empty(0, device=device)

def to_torch(*objs: Any) -> torch.Tensor:
    if len(objs) == 1:
        return _to_torch(REF_TENSOR, objs[0])
    return tuple(_to_torch(REF_TENSOR, x) for x in objs)


def _to_torch(ref_tensor: torch.Tensor, obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj.to(ref_tensor)
    return _to_torch(ref_tensor, torch.from_numpy(np.array(obj)))


def vec(*xs: Union[torch.Tensor, float]) -> torch.Tensor:
    if isinstance(xs[0], torch.Tensor):
        return torch.stack(xs, axis=-1)
    return REF_TENSOR.new_tensor(list(xs))


def torch_max(
    x1: Union[torch.Tensor, float], x2: Union[torch.Tensor, float]
) -> Union[torch.Tensor, float]:
    if not isinstance(x1, torch.Tensor) and not isinstance(x2, torch.Tensor):
        return max(x1, x2)
    elif not isinstance(x1, torch.Tensor):
        return x2.clamp(min=x1)
    elif not isinstance(x2, torch.Tensor):
        return x1.clamp(min=x2)
    return torch.maximum(x1, x2)


def torch_min(
    x1: Union[torch.Tensor, float], x2: Union[torch.Tensor, float]
) -> Union[torch.Tensor, float]:
    if not isinstance(x1, torch.Tensor) and not isinstance(x2, torch.Tensor):
        return min(x1, x2)
    elif not isinstance(x1, torch.Tensor):
        return x2.clamp(max=x1)
    elif not isinstance(x2, torch.Tensor):
        return x1.clamp(max=x2)
    return torch.minimum(x1, x2)


