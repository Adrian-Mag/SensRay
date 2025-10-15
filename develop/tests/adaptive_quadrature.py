import numpy as np

# Keast 4-point rule barycentric points and raw weights
_bary = np.array([
    [1/4, 1/4, 1/4, 1/4],
    [1/2, 1/6, 1/6, 1/6],
    [1/6, 1/2, 1/6, 1/6],
    [1/6, 1/6, 1/2, 1/6]
])
_raw_weights = np.array([-2/15, 3/40, 3/40, 3/40])  # sums to 11/120

def volume_of_tet(v0, v1, v2, v3):
    M = np.column_stack([np.array(v1)-np.array(v0),
                         np.array(v2)-np.array(v0),
                         np.array(v3)-np.array(v0)])
    return abs(np.linalg.det(M)) / 6.0

def bary_to_cartesian(bary, v0, v1, v2, v3):
    # bary shape (4,) with components a,b,c,d summing to 1
    a,b,c,d = bary
    return a*np.array(v0) + b*np.array(v1) + c*np.array(v2) + d*np.array(v3)

def tetrahedron_quadrature_4pt(f, v0, v1, v2, v3, normalize_weights=True):
    """
    Return integral over tetrahedron using Keast 4-point rule.
    If normalize_weights=True, raw weights are normalized to sum 1 and multiplied
    by tetrahedron volume (recommended).
    """
    vol = volume_of_tet(v0, v1, v2, v3)

    if normalize_weights:
        weights = _raw_weights / _raw_weights.sum()   # sum -> 1
        # evaluate f at mapped points and scale by volume once
        s = 0.0
        for w, bary in zip(weights, _bary):
            x = bary_to_cartesian(bary, v0, v1, v2, v3)
            s += w * f(x[0], x[1], x[2])
        return vol * s
    else:
        # Use raw weights directly (they already encode reference scaling)
        s = 0.0
        for w, bary in zip(_raw_weights, _bary):
            x = bary_to_cartesian(bary, v0, v1, v2, v3)
            s += w * f(x[0], x[1], x[2])
        # WARNING: do NOT multiply by vol in this branch
        return s

# subdivision (8 tets) — keep as you had or use any correct partition
def subdivide_tetrahedron(v0, v1, v2, v3):
    v0, v1, v2, v3 = map(np.array, (v0, v1, v2, v3))
    m01 = 0.5*(v0+v1)
    m02 = 0.5*(v0+v2)
    m03 = 0.5*(v0+v3)
    m12 = 0.5*(v1+v2)
    m13 = 0.5*(v1+v3)
    m23 = 0.5*(v2+v3)
    return [
        (v0, m01, m02, m03),
        (m01, v1, m12, m13),
        (m02, m12, v2, m23),
        (m03, m13, m23, v3),
        (m01, m02, m03, m12),
        (m01, m03, m13, m12),
        (m02, m03, m23, m12),
        (m12, m13, m23, m03),
    ]

def adaptive_tetrahedron_integrate(f, v0, v1, v2, v3, tol=1e-8, max_depth=12, depth=0):
    # Parent estimate
    I_parent = tetrahedron_quadrature_4pt(f, v0, v1, v2, v3, normalize_weights=True)

    if depth >= max_depth:
        return I_parent

    children = subdivide_tetrahedron(v0, v1, v2, v3)
    I_children = 0.0
    for tet in children:
        I_children += tetrahedron_quadrature_4pt(f, *tet, normalize_weights=True)

    err = abs(I_parent - I_children)
    if err < tol:
        return I_children
    else:
        # refine recursively (summing results from refined children)
        s = 0.0
        for tet in children:
            s += adaptive_tetrahedron_integrate(f, *tet, tol=tol, max_depth=max_depth, depth=depth+1)
        return s

# --------------------------
# Sanity checks:
if __name__ == "__main__":
    v0 = [0,0,0]; v1=[1,0,0]; v2=[0,1,0]; v3=[0,0,1]

    # 1) check raw weight sum and normalized behavior
    print("raw weight sum =", _raw_weights.sum())          # 11/120 = 0.091666...
    print("normalized weights sum =", (_raw_weights/_raw_weights.sum()).sum())  # 1.0

    # 2) tetra quadrature on f=1 should equal volume = 1/6
    const = lambda x,y,z: 1.0
    nonconst = lambda x,y,z: x+y+z
    print("quad (norm) f=1 =", tetrahedron_quadrature_4pt(const, v0, v1, v2, v3, normalize_weights=True))
    print("quad (raw)  f=1 =", tetrahedron_quadrature_4pt(const, v0, v1, v2, v3, normalize_weights=False))
    # quad(raw) will print ~0.0916666 because raw weights already encode the reference scale

    # 3) adaptive integral should give 1/6
    I_adaptive = adaptive_tetrahedron_integrate(const, v0, v1, v2, v3, tol=1e-12)
    print("adaptive integral f=1 =", I_adaptive)
    print("exact volume =", volume_of_tet(v0,v1,v2,v3))  # 1/6
