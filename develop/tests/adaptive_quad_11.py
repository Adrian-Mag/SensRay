import numpy as np
from itertools import permutations
from collections import defaultdict

# --------------------------
# Keast-11 barycentric rule
# --------------------------
def keast_11_barycentric():
    a = 11/14.0
    b = 1/14.0
    c = 1/4.0
    d = 1/2.0
    perm_abb = {tuple(p) for p in permutations([a,b,b,b])}   # 4 perms
    perm_ccdd = {tuple(p) for p in permutations([c,c,d,d])}  # 6 perms
    bary = [(0.25, 0.25, 0.25, 0.25)]
    weights = [-74.0/5625.0]
    bary.extend(list(perm_abb)); weights.extend([343.0/45000.0]*4)
    bary.extend(list(perm_ccdd)); weights.extend([56.0/2250.0]*6)
    return np.array(bary), np.array(weights)

_BARY, _WEIGHTS = keast_11_barycentric()

def bary_to_cartesian(lam, verts):
    # lam: length-4 barycentric tuple, verts: array([v0,v1,v2,v3])
    return lam[0]*verts[0] + lam[1]*verts[1] + lam[2]*verts[2] + lam[3]*verts[3]

def tet_volume(v0, v1, v2, v3):
    v0,v1,v2,v3 = map(np.asarray, (v0,v1,v2,v3))
    return abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0

def keast_11_integrate(f, v0, v1, v2, v3):
    """
    Evaluate integral approximation over tetrahedron using Keast-11 rule.
    Returns vol * sum_i w_i * f(x_i), i.e. includes the physical volume factor.
    """
    verts = np.array([v0, v1, v2, v3], dtype=float)
    vol = tet_volume(v0, v1, v2, v3)
    s = 0.0
    for lam, w in zip(_BARY, _WEIGHTS):
        p = bary_to_cartesian(lam, verts)
        s += w * float(f(float(p[0]), float(p[1]), float(p[2])))
    return vol * s

# --------------------------
# Subdivision (8 sub-tets)
# --------------------------
def subdivide_tetrahedron(v0, v1, v2, v3):
    v0,v1,v2,v3 = map(np.asarray, (v0,v1,v2,v3))
    m01 = 0.5*(v0 + v1)
    m02 = 0.5*(v0 + v2)
    m03 = 0.5*(v0 + v3)
    m12 = 0.5*(v1 + v2)
    m13 = 0.5*(v1 + v3)
    m23 = 0.5*(v2 + v3)
    return [
        (v0,  m01, m02, m03),
        (m01, v1,  m12, m13),
        (m02, m12, v2,  m23),
        (m03, m13, m23, v3),
        (m01, m02, m03, m12),
        (m01, m12, m13, m23),
        (m02, m12, m23, m03),
        (m03, m13, m23, m12),
    ]

# --------------------------
# Robust adaptive integrator
# --------------------------
def adaptive_tetrahedron_integrate(
    f,
    v0, v1, v2, v3,
    tol=1e-6,
    abs_tol=1e-12,
    max_depth=8,
    min_volume_frac=1e-6,
    max_nodes=20000,
    debug=False,
    print_every=500
):
    """
    Adaptive integrate f over tetrahedron (v0,v1,v2,v3).
    Returns (integral, stats).
    Robust stopping rules and EPS early-accept to avoid over-refinement.
    """
    # constants
    EPS_EARLY = 1e-14  # machine-epsilon-style early acceptance

    root_verts = (np.asarray(v0, dtype=float), np.asarray(v1, dtype=float),
                  np.asarray(v2, dtype=float), np.asarray(v3, dtype=float))
    root_vol = tet_volume(*root_verts)
    min_volume = max(root_vol * min_volume_frac, 1e-20)

    total_integral = 0.0
    nodes_processed = 0
    accepted = 0
    depth_hist = defaultdict(int)

    # stack: (v0,v1,v2,v3, depth, parent_parent_I) ; parent_parent_I used to detect non-improvement if desired
    stack = [(root_verts[0], root_verts[1], root_verts[2], root_verts[3], 0, None)]
    last_print = 0

    while stack:
        va, vb, vc, vd, depth, parent_parent_I = stack.pop()
        nodes_processed += 1

        # global node cap
        if nodes_processed > max_nodes:
            Ip = keast_11_integrate(f, va, vb, vc, vd)
            total_integral += Ip
            accepted += 1
            depth_hist[depth] += 1
            if debug and (accepted - last_print) >= print_every:
                print(f"[max_nodes] accepted {accepted} elements so far.")
                last_print = accepted
            continue

        vol = tet_volume(va, vb, vc, vd)
        I_parent = keast_11_integrate(f, va, vb, vc, vd)

        # Early accept on tiny volume or max depth
        if (vol <= min_volume) or (depth >= max_depth):
            total_integral += I_parent
            accepted += 1
            depth_hist[depth] += 1
            if debug and (accepted - last_print) >= print_every:
                print(f"[accept-vol/depth] depth={depth} vol={vol:.3e}")
                last_print = accepted
            continue

        # compute children estimates
        children = subdivide_tetrahedron(va, vb, vc, vd)
        child_vol_sum = sum(tet_volume(*ch) for ch in children)
        # safety: if partitioning bad, accept parent
        if not np.isfinite(child_vol_sum) or abs(child_vol_sum - vol) > 1e-12 * max(1.0, vol):
            total_integral += I_parent
            accepted += 1
            depth_hist[depth] += 1
            if debug:
                print(f"[warn] bad partition: child_vol_sum={child_vol_sum:.3e} parent_vol={vol:.3e}")
            continue

        I_children = 0.0
        for ch in children:
            I_children += keast_11_integrate(f, *ch)

        err = abs(I_parent - I_children)
        scale = abs(I_parent) + abs(I_children) + 1e-30
        rel_err = err / scale

        # --- EPS early accept: if difference is essentially machine-noise compared to parent magnitude ---
        if err < EPS_EARLY * max(1.0, abs(I_parent)):
            total_integral += I_children
            accepted += 1
            depth_hist[depth] += 1
            if debug and (accepted - last_print) >= print_every:
                print(f"[eps-accept] depth={depth} err={err:.3e}")
                last_print = accepted
            continue

        # normal stopping rules
        if (rel_err < tol) or (err < abs_tol):
            total_integral += I_children
            accepted += 1
            depth_hist[depth] += 1
            if debug and (accepted - last_print) >= print_every:
                print(f"[accept] depth={depth} rel_err={rel_err:.3e}")
                last_print = accepted
            continue
        else:
            # push children for further refinement
            for ch in children:
                stack.append((ch[0], ch[1], ch[2], ch[3], depth + 1, I_parent))

    stats = {
        "nodes_processed": nodes_processed,
        "accepted_elements": accepted,
        "root_volume": root_vol,
        "depth_histogram": dict(depth_hist)
    }
    if debug:
        print("Finished adaptive integration. Stats:", stats)
    return total_integral, stats

# --------------------------
# Quick test: integrate x+y+z over unit tetrahedron
# --------------------------
if __name__ == "__main__":
    f = lambda x,y,z: x + y + z
    v0 = (0.0, 0.0, 0.0)
    v1 = (1.0, 0.0, 0.0)
    v2 = (0.0, 1.0, 0.0)
    v3 = (0.0, 0.0, 1.0)

    I, stats = adaptive_tetrahedron_integrate(
        f, v0, v1, v2, v3,
        tol=1e-8,
        abs_tol=1e-14,
        max_depth=8,
        min_volume_frac=1e-8,
        max_nodes=20000,
        debug=True,
        print_every=10
    )
    print("Integral:", I)
    print("Stats:", stats)
