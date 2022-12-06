import torch
import matplotlib.pyplot as plt
from sdf import *

name = 'test'

def obj(sdf1: torch.Tensor, sdf2: torch.Tensor, lamda: torch.Tensor=None):
    return sdf1 + sdf2 + (sdf1 - sdf2)**2#.abs()

def main(sdf_func1, sdf_func2, max_iter=1000, lr=1e-2, lamda=None):
    # Get sample-set
    x_min, x_max = estimate_bounds(sdf_func1 | sdf_func2)
    Xs = np.linspace(x_min[0], x_max[0], 100)
    Ys = np.linspace(x_min[1], x_max[1], 200)
    Xs = [Xs, Ys]
    shape = [len(X) for X in Xs]
    P = cartesian_product(*Xs)
    P = tu.to_torch(P)

    # Prepare optimization variables
    x_min, x_max = tu.to_torch(x_min, x_max)
    N_init = 100
    cand_pts_init =  x_min + torch.rand(N_init, 2) * (x_max - x_min)
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
        print(f"{l :.3f} | {constr :.3f}")# | {lamda.mean().item() :.3f} | {lamda.std().item() :.3f}")


    # VISUALIZE
    cmap = plt.get_cmap('seismic')
    VMAX = 20
    # Prepare grid-mesh for vizualization
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
    ax5.imshow (my_obj.T, cmap, extent=extent, origin='lower', vmin=-VMAX, vmax=VMAX)
    ax5.contour(X, Y, sdf1_grid, [0], colors='k')#, extent=extent)
    ax5.contour(X, Y, sdf2_grid, [0], colors='k')#, extent=extent)
    ax5.contour(X, Y, my_obj, [-2, -1, 0, 1, 2], extent=extent)
    ax5.scatter(cand_pts_init[:, 0].detach(), cand_pts_init[:, 1].detach(), color = 'red')
    ax5.scatter(cand_pts[:, 0].detach(), cand_pts[:, 1].detach(), color='blue')
    # ax[0][1].scatter(P[:, 0].detach(), P[:, 1].detach(), color='gray', marker='x')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, my_obj, cmap=cmap, alpha=0.5, vmin=-VMAX, vmax=VMAX)
    ax.scatter(cand_pts[:, 0].detach(), cand_pts[:, 1].detach(), obj(sdf_func1(cand_pts), sdf_func2(cand_pts)).detach().numpy())
    ax.set_title("Objective: SDF1+SDF2 + (SDF1-SDF2)^2")
    plt.tight_layout()
    plt.show()
    # plt.savefig("SDF-SDF-contact.pdf")

    # print("P1 mean: {}, var: {}".format(p1.mean(dim=1), p1.var(dim=1)))
    # print("P2 mean: {}, var: {}".format(p2.mean(dim=1), p2.var(dim=1)))

if __name__ == "__main__":
    sdf_func1 = circle(1., center=[0.,2.])
    sdf_func2 = rectangle(1.3, center=[0.0, 4.0])
    sdf_func2 = circle(1.1, center=[0.0, 0.0])

    sdf_func3 = d2.equilateral_triangle().translate([2.,0]).circular_array(8)



    main(sdf_func2, sdf_func3, max_iter=500)



