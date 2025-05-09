from sdf import *
import math

# main body
f = rounded_cylinder(1, 0.1, 5)

# knurling
x = box((1, 1, 4)).rotate(math.pi / 4)
x = x.circular_array(24, 1.6)
x = x.twist(0.75) | x.twist(-0.75)
f -= x.k(0.1)

# central hole
f -= cylinder(0.5).k(0.1)

# vent holes
c = cylinder(0.25).orient(X)
f -= c.translate(Z * -2.5).k(0.1)
f -= c.translate(Z * 2.5).k(0.1)

if __name__ == "__main__":
    f.save('outputs/knurling.stl', samples=2**4)
