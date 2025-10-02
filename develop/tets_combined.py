# nlayer_sphere_sizes_smooth.py
# pip install gmsh meshio pyvista numpy

import gmsh, meshio, numpy as np, pyvista as pv

def build_nlayer_sphere_tets(
    radii,              # list[float], ascending, INCLUDE outer radius (e.g., [R1, R2, ..., Rn])
    H_layers,           # list[float], len == len(radii); target size for each layer
    W_trans,            # list[float], len == len(radii)-1; half-widths for transitions at radii[:-1]
    msh_out="nlayer.msh",
    vtu_out="nlayer.vtu",
    do_optimize=True,
):
    """
    Layers are: (0..R1], (R1..R2], ..., (R_{n-1}..R_n].  Smooth transitions around each R_i (i<n).
    """
    # ---- checks ----
    r = list(map(float, radii))
    assert all(r[i] < r[i+1] for i in range(len(r)-1)), "radii must be strictly ascending"
    assert len(H_layers) == len(r), "H_layers must have one value per layer"
    assert len(W_trans) == len(r)-1, "W_trans must have one value per internal interface"
    assert all(w > 0 for w in W_trans), "All W_trans must be > 0"

    R_out = r[-1]

    # ---- Gmsh init & options ----
    gmsh.initialize()
    gmsh.model.add("nlayer_sphere")

    # Make the background field authoritative (disable heuristics)
    for name in (
        "Mesh.MeshSizeFromPoints","Mesh.MeshSizeFromCurvature","Mesh.MeshSizeExtendFromBoundary",
        "Mesh.CharacteristicLengthFromPoints","Mesh.CharacteristicLengthFromCurvature","Mesh.CharacteristicLengthExtendFromBoundary",
    ):
        try: gmsh.option.setNumber(name, 0)
        except: pass
    for name in ("Mesh.MeshSizeMin","Mesh.MeshSizeMax","Mesh.CharacteristicLengthMin","Mesh.CharacteristicLengthMax"):
        try: gmsh.option.setNumber(name, 1e-9 if "Min" in name else 1e12)
        except: pass

    # ---- Geometry: outer sphere + all inner spheres ----
    tags_inner = []
    for Ri in r[:-1]:
        tags_inner.append(gmsh.model.occ.addSphere(0,0,0, Ri))
    tag_outer = gmsh.model.occ.addSphere(0,0,0, R_out)

    # Fragment so all interfaces are shared (conforming mesh)
    gmsh.model.occ.fragment([(3, tag_outer)], [(3, t) for t in tags_inner])
    gmsh.model.occ.synchronize()

    # ---- Physical groups for volumes: sorted by bounding radius (small->large) ----
    vols = gmsh.model.getEntities(3)  # [(3, tag), ...]
    def extent(tag):
        x0,y0,z0,x1,y1,z1 = gmsh.model.occ.getBoundingBox(3, tag)
        return max(abs(x0),abs(x1),abs(y0),abs(y1),abs(z0),abs(z1))
    vols_sorted = [t for (_, t) in sorted(vols, key=lambda e: extent(e[1]))]
    assert len(vols_sorted) == len(r), "Unexpected number of volumes after fragment"
    for i, vt in enumerate(vols_sorted, start=1):
        gmsh.model.addPhysicalGroup(3, [vt], i)
        gmsh.model.setPhysicalName(3, i, f"layer_{i-1}")

    # (optional) Physical groups for each spherical interface (handy for QC)
    surfs = gmsh.model.getEntities(2)
    areas = [gmsh.model.occ.getMass(2, s[1]) for s in surfs]
    for i, Ri in enumerate(r[:-1], start=0):
        # find surface with area ~ 4*pi*Ri^2
        idx = int(np.argmin([abs(a - 4*np.pi*Ri*Ri) for a in areas]))
        iface = surfs[idx][1]
        tag = 100 + i
        gmsh.model.addPhysicalGroup(2, [iface], tag)
        gmsh.model.setPhysicalName(2, tag, f"r={Ri}")

    # ---- Background size field: N-layer smooth blend (single MathEval expression) ----
    # Define s_k(r) = 0.5*(1+tanh((r-R_k)/W_k)) for k=0..n-2 (for radii[:-1])
    # Windows:
    #   w0      = (1 - s0)
    #   wi(1..n-2) = (prod_{k< i} s_k) * (1 - s_i)
    #   w_{n-1} = (prod_{k=0..n-2} s_k)
    r_expr = "sqrt(x*x+y*y+z*z)"
    s_exprs = [f"(0.5*(1+tanh(({r_expr}-{r[i]})/{W_trans[i]})))" for i in range(len(r)-1)]

    def prod_expr(seq):
        if not seq: return "1"
        out = seq[0]
        for e in seq[1:]:
            out = f"({out})*({e})"
        return out

    windows = []
    # layer 0
    windows.append(f"(1-({s_exprs[0]}))" if s_exprs else "1")
    # middle layers
    for i in range(1, len(r)-1):
        left = prod_expr(s_exprs[:i])
        windows.append(f"({left})*(1-({s_exprs[i]}))")
    # outermost layer
    if s_exprs:
        windows.append(prod_expr(s_exprs))
    # Build H(r) = sum_i H_i * window_i
    terms = [f"({float(H_layers[i])})*({windows[i]})" for i in range(len(H_layers))]
    H_expr = terms[0]
    for t in terms[1:]:
        H_expr = f"({H_expr})+({t})"

    fid = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(fid, "F", H_expr)
    gmsh.model.mesh.field.setAsBackgroundMesh(fid)

    # ---- Mesh ----
    gmsh.model.mesh.generate(3)
    if do_optimize:
        try:
            gmsh.model.mesh.optimize("Netgen")
        except:
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    gmsh.write(msh_out)
    gmsh.finalize()

    # ---- Convert to VTU ----
    mesh = meshio.read(msh_out)
    meshio.write(vtu_out, mesh)


if __name__ == "__main__":
    # EXAMPLE: PREM-like set (core, CMB, 660, Moho, surface)
    RADII   = [1221.5, 3480.0, 5701.0, 6331.0, 6371.0]  # include outer radius
    H_LAY   = [1000,  700,  500.0,  400.0,  100.0]   # one per layer
    W_TRANS = [200.0,  250.0,  200.0,  20.0]            # one per internal interface

    build_nlayer_sphere_tets(RADII, H_LAY, W_TRANS,
                             msh_out="nlayer.msh",
                             vtu_out="nlayer.vtu",
                             do_optimize=True)

    # ---- quick visualization (clip) ----
    grid = pv.read("nlayer.vtu")
    # Pull physical ids into a flat array named region_id
    region_ids = None
    for key in ("gmsh:physical","physical"):
        if key in grid.cell_data:
            d = grid.cell_data[key]
            if isinstance(d, np.ndarray) and d.size == grid.n_cells:
                region_ids = d; break
            if isinstance(d, (list,tuple)):
                for a in d:
                    if hasattr(a,"size") and a.size == grid.n_cells:
                        region_ids = a; break
            if region_ids is not None: break
    if region_ids is None:  # fallback by center radius
        rc = np.linalg.norm(grid.cell_centers().points, axis=1)
        bins = np.r_[0.0, RADII]
        region_ids = np.digitize(rc, bins, right=True)  # 1..n
    grid.cell_data["region_id"] = np.asarray(region_ids, dtype=np.int32)

    clip = grid.clip(normal=(1,0,0), origin=(0,0,0))
    p = pv.Plotter()
    p.add_mesh(grid.extract_surface(), color="lightgray", opacity=0.12)
    p.add_mesh(clip, scalars="region_id", cmap="tab10",
               show_edges=True, edge_color="black", interpolate_before_map=False)
    p.add_scalar_bar(title="layer id (1..N)")
    p.show()
