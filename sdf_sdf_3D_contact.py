import torch
import numpy as np
from sdf import d3, estimate_bounds, cartesian_product, tu, mesh
from examples import pawn, gearlike
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time


def unique(tensor1d):
    _, idx = np.unique(tensor1d.numpy(), return_inverse=True)
    return torch.from_numpy(idx)

def obj(sdf1: torch.Tensor, sdf2: torch.Tensor, lamda: torch.Tensor=None):
    return sdf1 + sdf2 + (sdf1 - sdf2)**2


def find_contacts_and_vis(sdf_func1, sdf_func2, max_iter=1000, lr=1e-2, n_cand=300):
    # Get sample-set
    x_min, x_max = estimate_bounds(sdf_func1 | sdf_func2)
    Xs = [np.linspace(x0, x1, 40) for x0, x1 in zip(x_min, x_max)]
    shape = [len(X) for X in Xs]
    P = cartesian_product(*Xs)
    P = tu.to_torch(P)


    # Prepare optimization variables
    x_min, x_max = tu.to_torch(x_min, x_max)
    cand_pts_init =  x_min + torch.rand(n_cand, 3) * (x_max - x_min)
    cand_pts = cand_pts_init.clone().requires_grad_(True)

    # opt = torch.optim.SGD([cand_pts], lr=lr)
    opt = torch.optim.Adam([cand_pts], lr=lr, weight_decay=1e-5)

    start = time.time()
    for i in range(max_iter):
        opt.zero_grad()
        sdfs1 = sdf_func1(cand_pts)
        sdfs2 = sdf_func2(cand_pts)
        sdf_vec = obj(sdfs1, sdfs2)

        loss = torch.sum(sdf_vec)
        loss.backward()
        opt.step()
        # with torch.no_grad():
            # lamda += lr * torch.sum(sdfs1 - sdfs2)
            # cand_pts -= lr * cand_pts.grad
            # lamda    += lr*3 * (sdfs1 - sdfs2).abs()
        # dx, dl = torch.autograd.grad(loss, [cand_pts, lamda])
        # with torch.no_grad():
            # cand_pts -= 1e-5 * dx
            # lamda += 1e-4 * (sdfs1 - sdfs2)
        # if i % 100 == 0:
        #     lamda = lamda+ 0.13

        # Log
        l = loss.item()
        constr = sum((sdfs1 - sdfs2).abs()).item()
        print(f"{i}: {l :.3f} | {constr :.3f} ({time.time() - start})")# | {lamda.mean().item() :.3f} | {lamda.std().item() :.3f}")


    # Extra stuff to present the results
    cand_pts = cand_pts.detach().numpy()

    # Get unique points from cand_pts (N, 3) -- also possible to use distances from (0,0,0) instead of pointwise comparison
    unique_points = np.unique(np.round(cand_pts, decimals=3), axis=0)
    n_unique = len(unique_points)

    print("We have {} unique points".format(n_unique))

    fig = plt.figure()
    ax = fig.add_subplot(projection=f'{3}d')

    cols = ['g', 'y']
    for sdf in [sdf_func1, sdf_func2]:
        points = mesh._worker(sdf, Xs, True)
        mymesh = Poly3DCollection(np.array(points).reshape(-1, 3, 3))
        mymesh.set_facecolor(cols.pop())
        mymesh.set_alpha(0.2)
        ax.add_collection3d(mymesh)

    # ax.scatter(cand_pts_init[:, 0], cand_pts_init[:, 1], cand_pts_init[:, 2], c='r', alpha=0.3, s=7, label=f'the {n_cand} initial points')
    # ax.scatter(cand_pts[:, 0], cand_pts[:, 1], cand_pts[:, 2], c='b', alpha=0.3, s=7, label='candidate points')
    ax.scatter(unique_points[:, 0], unique_points[:, 1], unique_points[:, 2], c='k', marker='x',s=42, label=f'the {n_unique} unique points')
    ax.set_xlim(left=x_min[0], right=x_max[0])
    ax.set_ylim(bottom=x_min[1], top=x_max[1])
    ax.set_zlim(x_min[2], x_max[2])
    ax.set_aspect('equal')

    plt.suptitle("3D-SDF contact detection")
    plt.legend()
    plt.show()


def pair1():
    return d3.box(a=[-12,-12,0.], b=[12,12,0.4]), d3.box(center=(0, 0, 0.75), size=4.5)

def pair2():
    return d3.box(size=2.), d3.sphere(center=(1.5, 0.0, 0.0), radius=0.5)

def pair3():
    return d3.box(size=2.).repeat(3, (1,0,0)), d3.sphere(center=(1.5, 0.0, 0.0), radius=0.5)

def pair4():
    f = gearlike.get()
    return f, d3.sphere(center=(2.3, 0.0, 0.0), radius=0.5)

def pair5():
    f1 = pawn.get()
    f2 = d3.box(center=(0.71, 0.0, 1.8), size=(1,1,2))
    return f1, f2

def pair4():
    f1 = gearlike.get()
    f2 = d3.sphere(center=(2.3, 0.0, 0.0), radius=0.9).circular_array(8, 0.2)
    return f1, f2

def pair5():
    return d3.box(size=(2., 5., 5)), d3.box(center=(2., 0., 0.), size=2.)

def pair6():
    return d3.box(size=(2., 5., 5), center=(0, 3.65, 0)), d3.capped_cylinder(-d3.Y*5, d3.Y*5, 0.5).rotate(np.pi/3, vector=d3.Z)

def pair7():
    return d3.box(center=(0., 0., 0), size=[10.,10.,0.5]), d3.box(center=(0., 0., 4.73), size=4.).rotate(np.pi/15., vector=d3.X).rotate(np.pi/16., vector=d3.Y)

if __name__ == "__main__":
    find_contacts_and_vis(*pair4(), 600, 1e-1, n_cand=300)
    find_contacts_and_vis(*pair1(), 600, 1e-1, n_cand=300)
    find_contacts_and_vis(*pair2(), 600, 1e-1, n_cand=300)
    find_contacts_and_vis(*pair3(), 600, 1e-1, n_cand=300)
    find_contacts_and_vis(*pair5(), 600, 1e-1, n_cand=300)
    find_contacts_and_vis(*pair6(), 600, 1e-1, n_cand=300)
    find_contacts_and_vis(*pair7(), 600, 1e-1, n_cand=300)
