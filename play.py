import matplotlib.pyplot as plt
from sdf import *

# f = circle(0.3, [0.5, 0.5])# | circle(0.3, [0.2, 0.2]))#.extrude(0.1) #| circle(0.2, [1.2, 1.2]).extrude(0.1)

# f.plot(samples=1000)
# f.save('outputs/play.obj')
# f.save()
# f.show_slice(z=0)


# f = rectangle(size=(0.2,0.5)).repeat((1,2), 1)#.extrude(0.1) #| circle(0.2, [1.2, 1.2]).extrude(0.1)



f = circle(0.2).repeat(1, (2,0)).extrude(0.4)

# Estimate box bounds of the sdf
bounds = estimate_bounds(f)

# Cube sampling
Xs = np.linspace(bounds[0], bounds[1], 10)
P = cartesian_product(*Xs.T)

# Visualize
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
aaa = ax.scatter(*P.T, c=f(tu.to_torch(P)).numpy(), alpha=0.4)
#ax.plot(*where.T, marker='^', color='r')
#ax.plot(*np.argwhere(sdf(tu.to_torch(P)).numpy() <= 1).T, marker='.', color='y')
plt.colorbar(aaa)
plt.show()
# f.plot('outputs/play.png')

# a = f(torch.rand(12222, 3))

# print(a.shape, type(a))