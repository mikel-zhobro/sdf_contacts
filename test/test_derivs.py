from sdf import *

problems = []
def test(f, name, samples=15, **kwargs):
    bounds = estimate_bounds(f, verbose=False)

    # Cube sampling
    Xs = np.linspace(bounds[0], bounds[1], 55)
    P = cartesian_product(*Xs.T)
    ixs = (f(tu.to_torch(P)).numpy() >= 0)[:,0]
    P = P[ixs]

    # Prepare samples
    P = tu.to_torch(P).requires_grad_(True)

    # Compute gradients
    val = f(P)
    ssum = val.sum()
    ssum.backward()
    grad = P.grad

    err = torch.norm(f(P-grad*val)).item()
    print(f"{name: <20}: {err: 0.8f}")

    if err > 1e-3:
        problems.append(name)


def main():
    # example
    f = sphere(1) & box(1.5)
    # c = cylinder(0.5)
    # f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    example = box(1)
    test(f, 'example')

    # sphere(radius=1, center=ORIGIN)
    f = sphere(1)
    test(f, 'sphere')

    # box(size=1, center=ORIGIN, a=None, b=None)
    f = box(1)
    test(f, 'box')

    f = box((1, 2, 3))
    test(f, 'box2')

    # rounded_box(size, radius)
    f = rounded_box((1, 2, 3), 0.25)
    test(f, 'rounded_box')

    # wireframe_box(size, thickness)
    f = wireframe_box((1, 2, 3), 0.05)
    test(f, 'wireframe_box')

    # torus(r1, r2)
    f = torus(1, 0.25)
    test(f, 'torus')

    # capsule(a, b, radius)
    f = capsule(-Z, Z, 0.5)
    test(f, 'capsule')

    # capped_cylinder(a, b, radius)
    f = capped_cylinder(-Z, Z, 0.5)
    test(f, 'capped_cylinder')

    # rounded_cylinder(ra, rb, h)
    f = rounded_cylinder(0.5, 0.1, 2)
    test(f, 'rounded_cylinder')

    # capped_cone(a, b, ra, rb)
    f = capped_cone(-Z, Z, 1, 0.5)
    test(f, 'capped_cone')

    # rounded_cone(r1, r2, h)
    f = rounded_cone(0.75, 0.25, 2)
    test(f, 'rounded_cone')

    # ellipsoid(size)
    f = ellipsoid((1, 2, 3))
    test(f, 'ellipsoid')

    # pyramid(h)
    f = pyramid(1)
    test(f, 'pyramid')

    # tetrahedron(r)
    f = tetrahedron(1)
    test(f, 'tetrahedron')

    # octahedron(r)
    f = octahedron(1)
    test(f, 'octahedron')

    # dodecahedron(r)
    f = dodecahedron(1)
    test(f, 'dodecahedron')

    # icosahedron(r)
    f = icosahedron(1)
    test(f, 'icosahedron')

    # plane(normal=UP, point=ORIGIN)
    f = sphere() & plane()
    test(f, 'plane')

    # slab(x0=None, y0=None, z0=None, x1=None, y1=None, z1=None, k=None)
    f = sphere() & slab(z0=-0.5, z1=0.5, x0=0)
    test(f, 'slab')

    # cylinder(radius)
    f = sphere() - cylinder(0.5)
    test(f, 'cylinder')

    # translate(other, offset)
    f = sphere().translate((0, 0, 2))
    test(f, 'translate')

    # scale(other, factor)
    f = sphere().scale((1, 2, 3))
    test(f, 'scale')

    # rotate(other, angle, vector=Z)
    # rotate_to(other, a, b)
    f = capped_cylinder(-Z, Z, 0.5).rotate(math.pi / 4, X)
    test(f, 'rotate')

    # orient(other, axis)
    c = capped_cylinder(-Z, Z, 0.25)
    f = c.orient(X) | c.orient(Y) | c.orient(Z)
    test(f, 'orient')

    # boolean operations

    a = box((3, 3, 0.5))
    b = sphere()

    # union
    f = a | b
    test(f, 'union')

    # difference
    f = a - b
    test(f, 'difference')

    # intersection
    f = a & b
    test(f, 'intersection')

    # smooth union
    f = a | b.k(0.25)
    test(f, 'smooth_union')

    # smooth difference
    f = a - b.k(0.25)
    test(f, 'smooth_difference')

    # smooth intersection
    f = a & b.k(0.25)
    test(f, 'smooth_intersection')

    # repeat(other, spacing, count=None, padding=0)
    f = sphere().repeat(3, (1, 1, 0))
    test(f, 'repeat')

    # circular_array(other, count, offset)
    f = capped_cylinder(-Z, Z, 0.5).circular_array(8, 4)
    test(f, 'circular_array')

    # blend(a, *bs, k=0.5)
    f = sphere().blend(box())
    test(f, 'blend')

    # dilate(other, r)
    f = example.dilate(0.1)
    test(f, 'dilate')

    # erode(other, r)
    f = example.erode(0.1)
    test(f, 'erode')

    # shell(other, thickness)
    f = sphere().shell(0.05) & plane(-Z)
    test(f, 'shell')

    # elongate(other, size)
    f = example.elongate((0.25, 0.5, 0.75))
    test(f, 'elongate')

    # twist(other, k)
    f = box().twist(math.pi / 2)
    test(f, 'twist')

    # bend(other, k)
    f = box().bend(1)
    test(f, 'bend')

    # bend_linear(other, p0, p1, v, e=ease.linear)
    f = capsule(-Z * 2, Z * 2, 0.25).bend_linear(-Z, Z, X, ease.in_out_quad)
    test(f, 'bend_linear')

    # bend_radial(other, r0, r1, dz, e=ease.linear)
    f = box((5, 5, 0.25)).bend_radial(1, 2, -1, ease.in_out_quad)
    test(f, 'bend_radial', sparse=False)

    # transition_linear(f0, f1, p0=-Z, p1=Z, e=ease.linear)
    f = box().transition_linear(sphere(), e=ease.in_out_quad)
    test(f, 'transition_linear')

    # transition_radial(f0, f1, r0=0, r1=1, e=ease.linear)
    f = box().transition_radial(sphere(), e=ease.in_out_quad)
    test(f, 'transition_radial')

    # extrude(other, h)
    f = hexagon(1).extrude(1)
    test(f, 'extrude')

    # extrude_to(a, b, h, e=ease.linear)
    f = rectangle(2).extrude_to(circle(1), 2, ease.in_out_quad)
    test(f, 'extrude_to')

    # revolve(other, offset=0)
    f = hexagon(1).revolve(3)
    test(f, 'revolve')

    # slice(other)
    f = example.translate((0, 0, 0.55)).slice().extrude(0.1)
    test(f, 'slice')

##############################################################################

if __name__ == "__main__":
    main()
    # f = sphere().scale((1, 2, 3))
    # test(f, 'scale')
    print([problems])

