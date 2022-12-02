import math

import torch
from matplotlib import pyplot as plt
from torch.nn.functional import normalize


y, x = torch.meshgrid(torch.arange(100), torch.arange(200))

c1 = torch.tensor([50., 50.])
c2 = torch.tensor([140., 50.])

r1 = torch.tensor([45.])
r2 = torch.tensor([45.])

circle_sdf = lambda pts, c, r: torch.sqrt((pts[0] - c[0]) ** 2 + (pts[1] - c[1]) ** 2) - r


def rect_sdf(pts_in, c, dims):
    pts = pts_in.reshape(2, -1) - c[:, None]
    half_dims = dims / 2

    dist = torch.abs(pts) - half_dims[:, None]
    signs = torch.sign(pts)
    signs[signs == 0] = 1

    max_dist, max_dist_ind = torch.max(dist, dim=0)

    q = torch.clamp(dist, min=0)

    sdfs = q.norm(dim=0) + torch.clamp(max_dist, max=0.)
    return sdfs


# sdf_func1 = lambda pts, c, dims: rect_sdf(lcp_physics.physics.utils.rotation_matrix(torch.tensor(math.pi / 5)) @ (pts.reshape(2, -1) - c[:, None]) + c[:, None], c, dims)
# sdf_func2 = lambda pts, c, dims: rect_sdf((pts.reshape(2, -1) - c[:, None]) + c[:, None], c, dims)
sdf_func1 = circle_sdf
sdf_func2 = circle_sdf

pts = torch.stack([x, y]).float()
sdf1 = sdf_func1(pts, c1, r1).reshape(100, 200)
sdf2 = sdf_func2(pts, c2, r2).reshape(100, 200)

cand_pts_init = torch.rand(2, 100) * torch.tensor([[200], [100]])
cand_pts = cand_pts_init.clone()
cand_pts.requires_grad = True

max_iter = 1500
lr = 1e-1
opt = torch.optim.SGD([cand_pts], lr=lr)

# fig, ax = plt.subplots(2, 2)
# cmap = plt.get_cmap('seismic')
# ax[0][0].imshow(sdf1, cmap, vmin=-100, vmax=100)
# ax[0][0].contour(sdf1, [-20, -10, 0, 10, 20])
# ax[0][0].set_title('SDF1')
# ax[1][0].imshow(sdf2, cmap, vmin=-100, vmax=100)
# ax[1][0].contour(sdf2, [-20, -10, 0, 10, 20])
# ax[1][0].set_title('SDF2')
# ax[0][1].imshow(sdf1 + sdf2, cmap, vmin=-100, vmax=100)
# ax[0][1].contour(sdf1 + sdf2, [-20, -10, 0, 10, 20])
# ax[0][1].scatter(cand_pts_init[0].detach(), cand_pts_init[1].detach())

for i in range(max_iter):
    opt.zero_grad()
    sdfs1 = sdf_func1(cand_pts, c1, r1)
    sdfs2 = sdf_func2(cand_pts, c2, r2)

    sum_sdf = torch.max(sdfs1, sdfs2) + sdfs1 + sdfs2
    loss = torch.sum(sum_sdf)
    loss.backward()
    opt.step()
    # cand_pts = cand_pts.detach() - normalize(cand_pts.grad, dim=1) * (sum_sdf.detach().abs()) * lr
    # cand_pts.requires_grad = True
    # cand_pts = cand_pts - sum_sdf * cand_pts.grad / cand_pts.grad.norm(dim=1, keepdim=True)

    # ax[0][1].scatter(cand_pts[0].detach(), cand_pts[1].detach())
    print(loss)

opt.zero_grad()
sdfs1 = sdf_func1(cand_pts, c1, r1)
loss1 = sdfs1.sum()
loss1.backward()
grad1 = cand_pts.grad.clone()

opt.zero_grad()
sdfs2 = sdf_func2(cand_pts, c2, r2)
loss2 = sdfs2.sum()
loss2.backward()
grad2 = cand_pts.grad.clone()

p1 = cand_pts - sdfs1 * grad1
p2 = cand_pts - sdfs2 * grad2

fig, ax = plt.subplots(2, 2)
cmap = plt.get_cmap('seismic')
ax[0][0].imshow(sdf1, cmap, vmin=-100, vmax=100)
ax[0][0].contour(sdf1, [-20, -10, 0, 10, 20])
ax[0][0].set_title('SDF1')
ax[1][0].imshow(sdf2, cmap, vmin=-100, vmax=100)
ax[1][0].contour(sdf2, [-20, -10, 0, 10, 20])
ax[1][0].set_title('SDF2')
ax[0][1].imshow(torch.max(sdf1, sdf2) + sdf1 + sdf2, cmap, vmin=-100, vmax=100)
ax[0][1].contour(torch.max(sdf1, sdf2) + sdf1 + sdf2, [-20, -10, 0, 10, 20])
ax[0][1].scatter(cand_pts_init[0].detach(), cand_pts_init[1].detach())
ax[0][1].scatter(cand_pts[0].detach(), cand_pts[1].detach())
ax[0][1].scatter(p1[0].detach(), p1[1].detach())
ax[0][1].scatter(p2[0].detach(), p2[1].detach())
ax[0][1].contour(sdf1, [0])
ax[0][1].contour(sdf2, [0])
ax[0][1].set_title("SDF1 + SDF2")
ax[1][1].imshow(torch.min(sdf1, sdf2), cmap, vmin=-100, vmax=100)
ax[1][1].contour(torch.min(sdf1, sdf2), [-20, -10, 0, 10, 20])
ax[1][1].set_title("min(SDF1, SDF2)")

fig2, ax2 = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
ax2[0].plot_surface(x.numpy(), y.numpy(), (sdf1 + sdf2).numpy(), cmap=cmap, vmin=-100, vmax=100)
ax2[1].plot_surface(x.numpy(), y.numpy(), (torch.max(sdf1, sdf2) + sdf1 + sdf2).numpy(), cmap=cmap, vmin=-100, vmax=100)

plt.show()
plt.tight_layout()
plt.savefig("SDF-SDF-contact.pdf")

print("P1 mean: {}, var: {}".format(p1.mean(dim=1), p1.var(dim=1)))
print("P2 mean: {}, var: {}".format(p2.mean(dim=1), p2.var(dim=1)))
