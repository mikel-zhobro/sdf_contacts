from functools import partial
from multiprocessing.pool import ThreadPool
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import multiprocessing
import itertools
import time
import numpy as np
import torch

from . import progress, stl, torch_util as tu

WORKERS = multiprocessing.cpu_count()
SAMPLES = 2 ** 6
BATCH_SIZE = 32

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def estimate_bounds(sdf, verbose=False):
    """
    ------ Estimate bounds of the sdf (dimension agnostic) ------
    starts with a small cube and expands it until sdf is contained
    it does that by expanding the bounds using sdf values from previous sampled bounds
    """
    n = sdf.dim
    s = 16
    c0 = np.zeros(n) - 1e9 #
    c1 = np.zeros(n) + 1e9 #
    prev = None
    for i in range(32):
        Cs = [np.linspace(x0, x1, s) for x0, x1 in zip(c0, c1)] # linspace for each dimension - (s,s,s) cubes [dim, s]
        d = np.abs(np.array([X[1] - X[0] for X in Cs])) # the diagonal of one of the (s,s,s) cubes
        threshold = np.linalg.norm(d) * 2 /3 # closer than the half of the diagonal is considered inside (changed to 2/3)
        # threshold = np.abs(d).mean() * 2 /3 # closer than the half of the diagonal is considered inside (changed to 2/3)
        if threshold == prev:
            break
        prev = threshold
        P = cartesian_product(*Cs) # shape: (s**n, n) where n can be 2 or 3 -- kind of meshgrid
        volume = sdf(tu.to_torch(P)).reshape(tuple([len(X) for X in Cs])) # (s, s, s) or (s, s)
        where = np.argwhere(np.abs(volume.numpy()) <= threshold+1)
        # where = np.argwhere(np.abs(volume.numpy()) <= threshold)

        c1 = c0 + where.max(axis=0) * d + d / 2
        c0 = c0 + where.min(axis=0) * d - d / 2
    if verbose:
        print(f"min/max coord: {where.max(axis=0)}, {where.min(axis=0)}")
        print(f"iteration {i} - threshold: {threshold} - c0: {c0} - c1: {c1}")
    # explain this function:
    #

    if verbose:
        print(c0, c1)
    return c0, c1

def estimate_bounds2_0(sdf, min=-10., max=10, verbose=True):
    """
    ------ Estimate bounds of the sdf (dimension agnostic) ------
    Same as above only that surface normals are used to expand the cube until the sdf is contained
    """
    n = sdf.dim
    s = 16
    c0 = np.zeros(n) - max #
    c1 = np.zeros(n) + max #
    prev = None
    print(f"iteration {0} - c0: {c0} - c1: {c1}")
    for i in range(1):
        Cs = [np.linspace(x0, x1, s) for x0, x1 in zip(c0, c1)] # linspace for each dimension - (s,s,s) cubes [dim, s]
        d = np.abs(np.array([X[1] - X[0] for X in Cs])) # the diagonal of one of the (s,s,s) cubes

        P = tu.to_torch(cartesian_product(*Cs)).requires_grad_() # shape: (s**n, n) where n can be 2 or 3 -- kind of meshgrid
        volume: torch.Tensor = sdf(P).reshape(tuple([len(X) for X in Cs])) # (s, s, s) or (s, s)
        volume.sum().backward()
        assert P.grad is not None
        vec_2_surface = np.abs(P.grad.numpy().reshape(tuple([len(X) for X in Cs]+[-1])) * volume.detach().numpy()[...,None]) # (s**n, n) * (s, s, s) = (s, s, s, n)

        where = np.argwhere(np.logical_and(vec_2_surface[:,:,0] <= d[0], vec_2_surface[:,:,1] <= d[1]))

        c1 = c0 + where.max(axis=0) * d + d / 2
        c0 = c0 + where.min(axis=0) * d - d / 2

        # print(f"min/max coord: {where.max(axis=0)}, {where.min(axis=0)}")
        print(f"iteration {i} - c0: {c0} - c1: {c1}, 'd': {d} {where.min(axis=0)}, {where.max(axis=0)}")
    # return np.where(c0 < min, min, c0), np.where(c1 > max, max, c0)
    return np.clip(c0-d-1., min, max), np.clip(c1+d+1., min, max)



def _worker(sdf, job, sparse):
    # Help functions for worker
    # (samples batches from job in the sdf according to sparse)
    n = sdf.dim
    def _marching_cubes(volume, level=0):
        verts, faces, _, _ = measure.marching_cubes(volume, level)
        return verts[faces].reshape((-1, 3))

    def _marching_squares(volume:np.ndarray, level=0):
        contours = measure.find_contours(volume, level)
        return np.concatenate(contours, axis=0)
        # return np.argwhere(volume <= 1e0)

    _marching = _marching_cubes if n == 3 else _marching_squares

    def _skip(sdf, job):
        # Skip if all points in job are either only inside or outside the sdf's 0 level
        # (this is a heuristic to skip batches that are not likely to contain surface points)
        x0 = np.array([X[0] for X in job])
        x1 = np.array([X[-1] for X in job])
        x = (x0 + x1).reshape(1, -1) / 2
        # sdf value in the center of the cube
        r = abs(sdf(tu.to_torch(x)).reshape(-1)[0]) # how far is the center from the 0 level-set
        # half of the diagonal of the cube
        d = np.linalg.norm(x.reshape(-1)-x0) # how far is the cube's corner from the center
        if r <= d:
            return False
        corners = np.array(list(itertools.product(*[(_x0, _x1) for _x0, _x1 in zip(x0, x1)])))
        values = sdf(tu.to_torch(corners)).reshape(-1).numpy()
        same = np.all(values > 0) if values[0] > 0 else np.all(values < 0)
        return same

    if sparse and n==3 and _skip(sdf, job):
        return None
        # return _debug_triangles(*job)
    P = cartesian_product(*job)
    volume = sdf(tu.to_torch(P)).numpy().reshape(tuple([len(X) for X in job]))
    try:
        points = points = _marching(volume)
    except Exception:
        return []

        # return _debug_triangles(*job)
    scale = np.array([X[1] - X[0] for X in job])
    offset = np.array([X[0] for X in job])
    return points * scale + offset

def generate(
    sdf,
    step=None, bounds=None, samples=SAMPLES,
    workers=WORKERS, batch_size=BATCH_SIZE,
    verbose=True, sparse=True):
    "Generate a mesh from a signed distance function in batched fashion."
    "returns a list of sorted 3d points where every 3 points form a triangle"
    "np.array(points, dtype='float32').reshape((-1, 3, 3)) can be used to get triangles"
    "sdf: a function that takes a numpy array of points and returns a numpy array of distances"
    start = time.time()
    n = sdf.dim

    if bounds is None:
        bounds = estimate_bounds(sdf)
    x0, x1 = bounds # x0: lower bounds (3,), x1: upper bounds (3,)

    if step is None:
        volume = np.prod(x1 - x0)
        step = (volume / (samples**n)) ** (1 / n)

    dx = np.zeros(n) # dx: step size (3,)
    dx[:] = step

    if verbose:
        print('min', *x0, sep=' ')
        print('max', *x1, sep=' ')
        print('step', *dx, sep=' ')

    X = [np.arange(x0_, x1_, dx_) for x0_, x1_, dx_ in zip(x0, x1, dx)]
    # we need only one batch for 2D, so we find only one contour
    s = batch_size if n==3 else len(X[0])
    Xs = [[X_t[i:i+s+1] for i in range(0, len(X_t), s)] for X_t in X]

    batches = list(itertools.product(*Xs))
    num_batches = len(batches)

    if verbose:
        num_samples = sum(np.prod([len(x_temp) for x_temp in xs]) for xs in batches)
        print('%d samples in %d batches with %d workers' %
            (num_samples, num_batches, workers))

    points = []
    skipped = empty = nonempty = 0
    bar = progress.Bar(num_batches, enabled=verbose)
    pool = ThreadPool(workers)
    f = partial(_worker, sdf, sparse=sparse)
    for result in pool.imap(f, batches):
        bar.increment(1)
        if result is None:
            skipped += 1
        elif len(result) == 0:
            empty += 1
        else:
            nonempty += 1
            points.extend(result)
    bar.done()

    seconds = time.time() - start
    if verbose:
        print('%d skipped, %d empty, %d nonempty' % (skipped, empty, nonempty))
        triangles = len(points) // 3
        print('%d triangles in %g seconds' % (triangles, seconds))

    return points, seconds

def save(path, *args, **kwargs):
    points, seconds = generate(*args, **kwargs)
    points = np.array(points)
    if len(points) > 0 and points.shape[1] == 2:
        n = points.shape[0] //4
        points = points[:n*4].reshape(-1, 2, 2)
        points = np.concatenate([points, np.sum(points, axis=1, keepdims=True)/2.], axis=1).reshape(-1, 2)
        points = np.concatenate([points, 1e-5+np.zeros((len(points), 1))], axis=1)
    if path.lower().endswith('.stl'):
        stl.write_binary_stl(path, points)
    else:
        mesh = _mesh(points)
        mesh.write(path)
    return seconds

def _mesh(points):
    import meshio
    points, cells = np.unique(points, axis=0, return_inverse=True)
    cells = [('triangle', cells.reshape((-1, 3)))]
    return meshio.Mesh(points, cells)

def plot(path=None, *args, **kwargs):
    sdf = args[0]
    n = sdf.dim
    bounds = estimate_bounds(sdf)
    points, seconds = generate(*args, **kwargs, bounds=bounds)

    if n ==3:
        mesh = Poly3DCollection(np.array(points).reshape(-1, 3, 3))
        mesh.set_facecolor('g')
        mesh.set_edgecolor('none')
        mesh.set_alpha(0.3)
        fig = plt.figure()
        ax = fig.add_subplot(projection=f'{n}d')
        ax.add_collection3d(mesh)
        ax.set_xlim(left=bounds[0][0], right=bounds[1][0])
        ax.set_ylim(bottom=bounds[0][1], top=bounds[1][1])
        ax.set_zlim(bounds[0][2], bounds[1][2])
    else:
        # fig, ax = plt.subplots()
        points = np.array(points).reshape(-1, 2)
        plt.plot(points[:, 1], points[:, 0], linewidth=2)
    # ax.view_init(20, -45)
    if path is not None:
        plt.savefig(path)
    # else:
    #     plt.show()
# --------------------------------------------------------------------
# Plot Slice
# --------------------------------------------------------------------
def sample_slice(
    sdf, w=1024, h=1024,
    x=None, y=None, z=None, bounds=None):

    if bounds is None:
        bounds = estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds

    if x is not None:
        X = np.array([x])
        Y = np.linspace(y0, y1, w)
        Z = np.linspace(z0, z1, h)
        extent = (Z[0], Z[-1], Y[0], Y[-1])
        axes = 'ZY'
    elif y is not None:
        Y = np.array([y])
        X = np.linspace(x0, x1, w)
        Z = np.linspace(z0, z1, h)
        extent = (Z[0], Z[-1], X[0], X[-1])
        axes = 'ZX'
    elif z is not None:
        Z = np.array([z])
        X = np.linspace(x0, x1, w)
        Y = np.linspace(y0, y1, h)
        extent = (Y[0], Y[-1], X[0], X[-1])
        axes = 'YX'
    else:
        raise Exception('x, y, or z position must be specified')

    P = cartesian_product(X, Y, Z)
    return sdf(tu.to_torch(P)).reshape((w, h)), extent, axes

def show_slice(*args, **kwargs):
    import matplotlib.pyplot as plt
    show_abs = kwargs.pop('abs', False)
    a, extent, axes = sample_slice(*args, **kwargs)
    if show_abs:
        a = np.abs(a)
    print(min(a.flatten()), max(a.flatten()))
    im = plt.imshow(a, extent=extent, origin='lower')
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    plt.colorbar(im)
    plt.show()



def _debug_triangles(X, Y, Z):
    x0, x1 = X[0], X[-1]
    y0, y1 = Y[0], Y[-1]
    z0, z1 = Z[0], Z[-1]

    p = 0.25
    x0, x1 = x0 + (x1 - x0) * p, x1 - (x1 - x0) * p
    y0, y1 = y0 + (y1 - y0) * p, y1 - (y1 - y0) * p
    z0, z1 = z0 + (z1 - z0) * p, z1 - (z1 - z0) * p

    v = [
        (x0, y0, z0),
        (x0, y0, z1),
        (x0, y1, z0),
        (x0, y1, z1),
        (x1, y0, z0),
        (x1, y0, z1),
        (x1, y1, z0),
        (x1, y1, z1),
    ]

    return [
        v[3], v[5], v[7],
        v[5], v[3], v[1],
        v[0], v[6], v[4],
        v[6], v[0], v[2],
        v[0], v[5], v[1],
        v[5], v[0], v[4],
        v[5], v[6], v[7],
        v[6], v[5], v[4],
        v[6], v[3], v[7],
        v[3], v[6], v[2],
        v[0], v[3], v[2],
        v[3], v[0], v[1],
    ]
