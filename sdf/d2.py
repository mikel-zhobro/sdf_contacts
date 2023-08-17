import functools
import math
from typing import Callable
import torch
from . import dn, d3, ease, mesh, torch_util as tu


# Constants

ORIGIN = torch.tensor((0,0), device=tu.device)

X = torch.tensor((1.,0.), device=tu.device)
Y = torch.tensor((0.,1.), device=tu.device)

UP = Y

# SDF Class

_ops = {}

class SDF2:
    dim = 2
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]):
        self.f = f
    def __call__(self, p):
        return self.f(p).reshape((-1, 1))
    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
        raise AttributeError
    def __or__(self, other):
        return union(self, other)
    def __and__(self, other):
        return intersection(self, other)
    def __sub__(self, other):
        return difference(self, other)
    def k(self, k=None):
        self._k = k
        return self
    def generate(self, *args, **kwargs):
        return mesh.generate(self, *args, **kwargs)
    def save(self, path, *args, **kwargs):
        return mesh.save(path, self, *args, **kwargs)
    def plot(self, path=None, *args, **kwargs):
        return mesh.plot(path, self, *args, **kwargs)

    # Operators
    def translate(self, offset): return translate(self, offset)
    def scale(self, factor): return scale(self, factor)
    def rotate(self, angle): return rotate(self, angle)
    def rotate_tensor(self, angle): return rotate_tensor(self, angle)
    def circular_array(self, count): return circular_array(self, count)
    def elongate(self, size): return elongate(self, size)

    # 2D => 3D
    def extrude(self, h): return extrude(self, h)
    def extrude_to(self, b, h, e=ease.linear): return extrude_to(self, b, h, e)
    def revolve(self, offset=0): return revolve(self, offset)

    # Common
    def union(self, *bs, k=None): return union(self, *bs, k=k)
    def difference(self, *bs, k=None): return difference(self, *bs, k=k)
    def intersection(self, *bs, k=None): return intersection(self, *bs, k=k)
    def blend(self, *bs, k=0.5): return blend(self, *bs, k=k)

    def negate(self): return negate(self)
    def dilate(self, r): return dilate(self, r)
    def erode(self, r): return erode(self, r)
    def shell(self, thickness): return shell(self, thickness)
    def repeat(self, spacing, count=None, padding=0): return repeat(self, spacing, count, padding)


# ----------------------------------------------------------------
# Definitions
# ----------------------------------------------------------------
def sdf2(f: Callable[..., SDF2]) -> Callable[..., SDF2]:
    def wrapper(*args, **kwargs):
        return SDF2(f(*args, **kwargs))
    return wrapper

def op2(f: SDF2) -> Callable[..., SDF2]:
    def wrapper(*args, **kwargs):
        return SDF2(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

def op23(f: SDF2):
    def wrapper(*args, **kwargs):
        return d3.SDF3(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def _length(a):
    return torch.linalg.norm(a, dim=1)

def _normalize(a):
    return a / torch.norm(a)

def _dot(a, b):
    return (a * b).sum(1)

_vec = tu.vec
_min = tu.torch_min
_max = tu.torch_max

# ----------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------
@sdf2
def circle(radius=1, center=ORIGIN):
    def f(p):
        return _length(p - tu.to_torch(center)) - radius
    return f

@sdf2
def line(normal=UP, point=ORIGIN):
    _normal = _normalize(tu.to_torch(normal))
    _point = tu.to_torch(point)

    def f(p):
        return torch.mv(p - _point, _normal)
    return f

@sdf2
def slab(x0=None, y0=None, x1=None, y1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(line(X, (x0, 0)))
    if x1 is not None:
        fs.append(line(-X, (x1, 0)))
    if y0 is not None:
        fs.append(line(Y, (0, y0)))
    if y1 is not None:
        fs.append(line(-Y, (0, y1)))
    return intersection(*fs, k=k)

@sdf2
def rectangle(size=1, center=ORIGIN, a=None, b=None):
    if a is not None and b is not None:
        a = tu.to_torch(a)
        b = tu.to_torch(b)
        size = b - a
        center = a + size / 2
        return rectangle(size, center)
    size, center = tu.to_torch(size, center)
    def f(p):
        q = (p - center).abs() - size
        return _length(_max(q, 0)) + _min(q.amax(1), 0)
    return f

@sdf2
def rounded_rectangle(size, radius, center=ORIGIN):
    try:
        r0, r1, r2, r3 = radius
    except TypeError:
        r0 = r1 = r2 = r3 = radius

    _size, _center = tu.to_torch(size, center)
    def f(p):
        x = p[:,0]
        y = p[:,1]
        r = torch.zeros_like(x)
        r[torch.logical_and(x > 0, y > 0)] = r0
        r[torch.logical_and(x > 0, y <= 0)] = r1
        r[torch.logical_and(x <= 0, y <= 0)] = r2
        r[torch.logical_and(x <= 0, y > 0)] = r3
        q = p.abs() - _size / 2 + r[:, None]
        return (
            _min(_max(q[:,0], q[:,1]), 0).reshape((-1, 1)) +
            _length(_max(q, 0)).reshape((-1, 1)) - r[:, None])
    return f

@sdf2
def equilateral_triangle():
    def f(p):
        k = 3 ** 0.5
        p = _vec(p[:, 0].abs() - 1, p[:, 1] + 1 / k)
        w = p[:,0] + k * p[:,1] > 0
        q = _vec(
            p[:,0] - k * p[:,1],
            -k * p[:,0] - p[:,1]) / 2
        p = torch.where(w.reshape((-1, 1)), q, p)
        p = _vec(p[:, 0] - p[:, 0].clamp(-2, 0), p[:, 1])
        return -_length(p) * torch.sign(p[:,1])
    return f

@sdf2
def hexagon(r):
    r *= 3 ** 0.5 / 2
    def f(p):
        r_torch = tu.to_torch(r)
        k = tu.to_torch([3 ** 0.5 / -2, 0.5, math.tan(math.pi / 6)])
        p0 = p.abs()
        p1 = p0 - 2 * k[:2] * _min(_dot(k[:2], p0), 0).reshape((-1, 1))
        p2 = p1 - _vec(p1[:,0].clamp(-k[2] * r_torch, k[2] * r_torch),torch.zeros_like(p1[:,0]) + r_torch)
        return _length(p2) * torch.sign(p2[:,1])
    return f

@sdf2
def rounded_x(w, r):
    def f(p):
        w, r = tu.to_torch(w, r)
        p = p.abs()
        q = (_min(p[:,0] + p[:,1], w) * 0.5).reshape((-1, 1))
        return _length(p - q) - r
    return f

@sdf2
def polygon(points): # TODO: fix
    points = [tu.to_torch(point) for point in points]
    def f(p):
        n = len(points)
        d = _dot(p - points[0], p - points[0])
        s = torch.ones_like(p[:, 0])
        for i in range(n):
            j = (i + n - 1) % n
            vi = points[i]
            vj = points[j]
            e = vj - vi
            w = p - vi
            b = w - e * (torch.mv(w, e) / torch.dot(e, e)).clamp(0, 1).reshape((-1, 1))
            d = _min(d, _dot(b, b))
            c1 = p[:,1] >= vi[1]
            c2 = p[:,1] < vj[1]
            c3 = e[0] * w[:,1] > e[1] * w[:,0]
            c = _vec(c1, c2, c3)
            s = torch.where(torch.all(c, axis=1) | torch.all(~c, axis=1), -s, s)
        return s * d.sqrt()
    return f

# ----------------------------------------------------------------
# Positioning
# ----------------------------------------------------------------
@op2
def translate(other, offset):
    def f(p):
        return other(p - tu.to_torch(offset))
    return f

@op2
def rotate(other, angle):
    s = math.sin(angle)
    c = math.cos(angle)
    matrix = tu.to_torch([
        [c, -s],
        [s, c],
    ])
    def f(p):
        return other(p@matrix)
    return f

@op2
def rotate_tensor(other, angle):
    _angle = tu.to_torch(angle)
    s = torch.sin(_angle)
    c = torch.cos(_angle)
    rot = torch.stack([torch.stack([c, -s]),
                       torch.stack([s, c])])
    def f(p):
        return other(p@rot)
    return f

@op2
def scale(other, factor):
    try:
        x, y = factor
    except TypeError:
        x = y = factor
    s = (x, y)
    m = min(x, y)
    def f(p):
        return other(p / tu.to_torch(s)) * tu.to_torch(m)
    return f



@op2
def circular_array(other, count):
    angles = [i / count * 2 * math.pi for i in range(count)]
    return union(*[other.rotate(a) for a in angles])

# ----------------------------------------------------------------
# Alterations
# ----------------------------------------------------------------
@op2
def elongate(other, size):
    def f(p):
        q = p.abs() - tu.to_torch(size)
        x = q[:,0].reshape((-1, 1))
        y = q[:,1].reshape((-1, 1))
        w = _min(_max(x, y), 0)
        return other(_max(q, 0)) + w
    return f

# ----------------------------------------------------------------
# 2D => 3D Operations
# ----------------------------------------------------------------
@op23
def extrude(other, h):
    def f(p):
        d = other(p[:,[0,1]])
        w = _vec(d.reshape(-1), p[:, 2].abs() - h / 2)
        return _min(_max(w[:,0], w[:,1]), 0) + _length(_max(w, 0))
    return f

@op23
def extrude_to(a, b, h, e=ease.linear):
    def f(p):
        d1 = a(p[:,[0,1]])
        d2 = b(p[:,[0,1]])
        t = e((p[:, 2] / h).clamp(-0.5, 0.5) + 0.5)
        d = d1 + (d2 - d1) * t.reshape((-1, 1))
        w = _vec(d.reshape(-1), p[:, 2].abs() - h / 2)
        return _min(_max(w[:,0], w[:,1]), 0) + _length(_max(w, 0))
    return f

@op23
def revolve(other, offset=0):
    def f(p):
        xy = p[:,[0,1]]
        q = _vec(_length(xy) - offset, p[:,2])
        return other(q)
    return f

# ----------------------------------------------------------------
# Common
# ----------------------------------------------------------------
union = op2(dn.union)
difference = op2(dn.difference)
intersection = op2(dn.intersection)
blend = op2(dn.blend)
negate = op2(dn.negate)
dilate = op2(dn.dilate)
erode = op2(dn.erode)
shell = op2(dn.shell)
repeat = op2(dn.repeat)
