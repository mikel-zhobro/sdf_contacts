from sdf import *

def get():
    f = sphere(2) & slab(z0=-0.5, z1=0.5).k(0.1)
    f -= cylinder(1).k(0.1)
    f -= cylinder(0.25).circular_array(16, 2).k(0.1)
    return f

if __name__ == "__main__":
    f = get()
    f.save('outputs/gearlike.stl', samples=2**2)
