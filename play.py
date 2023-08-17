import torch
import matplotlib.pyplot as plt
import numpy as np


from sdf import *

name = 'test'


def visualize(cand_pts_init, cand_pts, unique_points, sdf_func1, sdf_func2, obj, x_min, x_max):
    n_cand = cand_pts.shape[0]
    n_unique = unique_points.shape[0]
    # VISUALIZE
    cmap = plt.get_cmap('seismic')
    VMAX = 20

    # Prepare grid-mesh for vizualization
    Xs = np.linspace(x_min[0], x_max[0], 100)
    Ys = np.linspace(x_min[1], x_max[1], 200)
    Xs = [Xs, Ys]
    shape = [len(X) for X in Xs]
    P = tu.to_torch(cartesian_product(*Xs))

    sdf1_vals = sdf_func1(P)
    sdf2_vals = sdf_func2(P)
    my_obj = obj(sdf1_vals, sdf2_vals).detach().numpy().reshape(shape)
    sdf1_grid = sdf1_vals.reshape(shape)
    sdf2_grid = sdf2_vals.reshape(shape)
    X = P[:,0].reshape(shape)
    Y = P[:,1].reshape(shape)
    extent = x_min[0], x_max[0], x_min[1], x_max[1]
    extent = x_min[0], x_max[0], x_min[1], x_max[1]


    fig = plt.figure(figsize=(10, 13))
    gs1 = fig.add_gridspec(nrows=3, ncols=2)
    ax1 = fig.add_subplot(gs1[0, 0])
    ax2 = fig.add_subplot(gs1[0, 1])
    ax3 = fig.add_subplot(gs1[1, 0])
    ax4 = fig.add_subplot(gs1[1, 1])
    ax5 = fig.add_subplot(gs1[2, :])


    # SDF1 and SDF2
    ax1.contour(X, Y, sdf1_grid, [0], colors='k')#, extent=extent)
    ax1.imshow(sdf1_grid.T, cmap, origin='lower', extent=extent, vmin=-VMAX, vmax=VMAX)
    ax1.set_title('SDF1')
    ax2.imshow(sdf2_grid.T, cmap, origin='lower', extent=extent, vmin=-VMAX, vmax=VMAX)
    ax2.contour(X, Y, sdf2_grid, [0], colors='k')#, extent=extent)
    ax2.set_title('SDF2')

    # SDF1 - SDF2
    SS = (sdf1_grid-sdf2_grid).abs()
    ax3.set_title("(SDF1-SDF2)^2")
    ax3.imshow(SS.T, cmap, extent=extent, origin='lower', vmin=-VMAX, vmax=VMAX)
    ax3.contour(X, Y, SS, [-2, -1, 0, 1, 2], extent=extent)
    ax3.contour(X, Y, sdf1_grid, [0], colors='k')#, extent=extent)
    ax3.contour(X, Y, sdf2_grid, [0], colors='k')#, extent=extent)

    # SDF1 + SDF2
    SS = sdf1_grid+sdf2_grid
    ax4.set_title("SDF1+SDF2")
    ax4.imshow(SS.T, cmap, extent=extent, origin='lower', vmin=-VMAX, vmax=VMAX)
    ax4.contour(X, Y, SS, [-4, -3, 0, 3, 4], extent=extent)
    ax4.contour(X, Y, sdf1_grid, [0], colors='k')#, extent=extent)
    ax4.contour(X, Y, sdf2_grid, [0], colors='k')#, extent=extent)

    # Objective
    ax5.set_title("Objective: SDF1+SDF2 + (SDF1-SDF2)^2")
    ax5.imshow (my_obj.T, cmap, extent=extent, origin='lower', vmin=-VMAX, vmax=VMAX, alpha=0.7)
    ax5.contour(X, Y, sdf1_grid, [0], colors='k')#, extent=extent)
    ax5.contour(X, Y, sdf2_grid, [0], colors='k')#, extent=extent)
    ax5.contour(X, Y, my_obj, [-2, -1, 0, 1, 2], extent=extent)
    ax5.scatter(cand_pts_init[:, 0], cand_pts_init[:, 1], c='r', s=7, label=f'the {n_cand} initial points')
    ax5.scatter(cand_pts[:, 0], cand_pts[:, 1], c='b', s=7, label='candidate points')
    ax5.scatter(unique_points[:, 0], unique_points[:, 1], c='k', marker='x', s=42, label=f'the {n_unique} unique points')

    plt.suptitle("2D-SDF contact detection")
    plt.legend()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, my_obj, cmap=cmap, alpha=0.5, vmin=-VMAX, vmax=VMAX)
    ax.scatter(cand_pts[:, 0], cand_pts[:, 1], obj(sdf_func1(tu.to_torch(cand_pts)), sdf_func2(tu.to_torch(cand_pts))).detach().numpy())
    ax.set_title("Objective: SDF1+SDF2 + (SDF1-SDF2)^2")
    plt.tight_layout()
    plt.show()


def find_contact(sdf_func1, sdf_func2, cand_pts_init: torch.Tensor, max_iter=1000, lr=1e-2, n_cand=100, ):

    cand_pts = cand_pts_init.clone().requires_grad_(True)

    opt = torch.optim.SGD([cand_pts], lr=lr)

    for i in range(max_iter):
        opt.zero_grad()
        sdfs1 = sdf_func1(cand_pts)
        sdfs2 = sdf_func2(cand_pts)
        sdf_vec = obj(sdfs1, sdfs2)

        loss = torch.sum(sdf_vec)
        loss.backward()
        opt.step()

        # Log
        l = loss.item()
        constr = sum((sdfs1 - sdfs2).abs()).item()
        print(f"{i}:{l :.3f} | {constr :.3f}")# | {lamda.mean().item() :.3f} | {lamda.std().item() :.3f}")

    return cand_pts

def obj(sdf1: torch.Tensor, sdf2: torch.Tensor):
    return sdf1 + sdf2 + (sdf1 - sdf2)**2 #.abs()



def estimate_bounds2_0(sdf, min=-10., max=10, verbose=True):
    """
    ------ Estimate bounds of the sdf (dimension agnostic) ------
    starts with a small cube and expands it until sdf is contained
    it does that by expanding the bounds using sdf values from previous sampled bounds
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


def main(sdf_func1, sdf_func2, max_iter=1000, lr=1e-2, n_cand=100, ):
    # Get initial sample-set
    x_min, x_max = estimate_bounds2_0(sdf_func1 | sdf_func2, min=(-10,-10), max=(10,10) )
    x_min, x_max = tu.to_torch(x_min, x_max)
    cand_pts_init =  x_min + torch.rand(n_cand, 2) * (x_max - x_min)

    contact_points = find_contact(sdf_func1, sdf_func2, cand_pts_init, max_iter=max_iter, lr=lr, n_cand=n_cand)


    # Extra stuff to present the results
    cand_pts_np = contact_points.detach().numpy()

    # Get unique points from cand_pts (N, 2) -- also possible to use distances from (0,0,0) instead of pointwise comparison
    unique_points = np.unique(np.round(cand_pts_np, decimals=3), axis=0)
    print("We have {} unique points".format(len(unique_points)))

    visualize(cand_pts_init.detach().numpy(), cand_pts_np, unique_points, sdf_func1, sdf_func2, obj, x_min, x_max)


if __name__ == "__main__":
    plane_sdf = line(normal=[0., 1.], point=[0., 0.])
    sdf_func1 = circle(1., center=[0.,0.])
    sdf_func2 = rectangle(0.3, center=[0.0, 0.0])
    sdf_func3 = circle(1.1, center=[0.0, 0.0])

    sdf_func4 = d2.equilateral_triangle().translate([2.,0]).circular_array(8)

    x = torch.tensor([1.0, 0.1])[None, :]
    print(plane_sdf(x))
    main(sdf_func2, sdf_func4, max_iter=120, lr=0.1, n_cand=100)



