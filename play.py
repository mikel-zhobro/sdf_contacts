from sdf.d2 import circle




f = (circle(0.3, [0.5, 0.5]) | circle(0.3, [0.2, 0.2]))#.extrude(0.1) | circle(0.2, [1.2, 1.2]).extrude(0.1)
# f.plot(samples=1000)
f.save('play.obj')
# f.plot()
# f.show_slice(z=0)