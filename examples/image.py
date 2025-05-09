from sdf import *

IMAGE = 'examples/butterfly.png'

w, h = measure_image(IMAGE)

f = rounded_box((w * 1.1, h * 1.1, 0.1), 0.05)
f &= image(IMAGE).extrude(1) & slab(z0=0, z1=0.075)

if __name__ == "__main__":
    f.save('outputs/image.stl')
