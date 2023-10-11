import itertools
import numpy as np
import torch
from . import torch_util as tu
_min = tu.torch_min
_max = tu.torch_max

def union(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, '_k', None)
            if K is None:
                d1 = _min(d1, d2)
            else:
                h = torch.clamp(0.5 + 0.5 * (d2 - d1) / K, 0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m - K * h * (1 - h)
        return d1
    return f

def difference(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, '_k', None)
            if K is None:
                d1 = _max(d1, -d2)
            else:
                h = torch.clamp(0.5 - 0.5 * (d2 + d1) / K, 0, 1)
                m = d1 + (-d2 - d1) * h
                d1 = m + K * h * (1 - h)
        return d1
    return f

def intersection(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, '_k', None)
            if K is None:
                d1 = _max(d1, d2)
            else:
                h = torch.clamp(0.5 - 0.5 * (d2 - d1) / K, 0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m + K * h * (1 - h)
        return d1
    return f

def blend(a, *bs, k=0.5):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, '_k', None)
            d1 = K * d2 + (1 - K) * d1
        return d1
    return f

def negate(other):
    def f(p):
        return -other(p)
    return f

def dilate(other, r):
    def f(p):
        return other(p) - r
    return f

def erode(other, r):
    def f(p):
        return other(p) + r
    return f

def shell(other, thickness):
    def f(p):
        return torch.abs(other(p)) - thickness / 2
    return f

def repeat(other, spacing, count=None, padding=0):
    if count is not None:
        count = tu.to_torch(count)
        assert count.numel() == other.dim, f"make sure count({count}) matches the dimension of the sdf({other.dim})"
    spacing = tu.to_torch(spacing)

    def neighbors(dim, padding, spacing):
        try:
            padding = [padding[i] for i in range(dim)]
        except Exception:
            padding = [padding] * dim
        try:
            spacing = [spacing[i] for i in range(dim)]
        except Exception:
            spacing = [spacing] * dim
        for i, s in enumerate(spacing):
            if s == 0:
                padding[i] = 0
        axes = [list(range(-p, p + 1)) for p in padding]
        return list(itertools.product(*axes))

    def f(p: torch.Tensor):
        q = torch.where(spacing != 0, p/spacing, torch.zeros_like(p))
        if count is None:
            index = torch.round(q)
        else:
            index = torch.clamp(torch.round(q), -count, count)

        indexes = [index + tu.to_torch(n) for n in neighbors(p.shape[-1], padding, spacing)]
        A = [other(p - tu.to_torch(spacing * i)) for i in indexes]
        a = A[0]
        for b in A[1:]:
            a = _min(a, b)
        return a
    return f