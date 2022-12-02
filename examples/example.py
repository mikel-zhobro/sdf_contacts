from sdf import *

f = sphere(1) & box(1.5)

c = cylinder(0.5)
f -= c.orient(X) | c.orient(Y) | c.orient(Z)


if __name__ == "__main__":
    f.save('outputs/out.stl')