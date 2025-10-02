# three_layer_sphere_sizes_smooth_fixed.py
# pip install gmsh meshio pyvista numpy

import gmsh, meshio, numpy as np, pyvista as pv

# ---------- parameters (km) ----------
R1 = 1221.5       # core radius
R2 = 3480.0       # mid-shell radius (e.g., CMB)
R3 = 6371.0       # outer radius (surface)

H_CORE  = 1000.0   # target tet size deep in core      (r << R1)
H_MID   = 500.0   # target tet size in middle shell   (R1 << r << R2)
H_SHELL = 250.0   # target tet size in outer shell    (r >> R2)

W1 = 200.0        # transition half-width around R1 (km)
W2 = 300.0        # transition half-width around R2 (km)

# ---------- Gmsh: CAD + fields + mesh ----------
gmsh.initialize()
gmsh.model.add("three_layer_sphere")

# Let the background field fully control sizes (disable heuristics)
for name in (
    "Mesh.MeshSizeFromPoints","Mesh.MeshSizeFromCurvature","Mesh.MeshSizeExtendFromBoundary",
    "Mesh.CharacteristicLengthFromPoints","Mesh.CharacteristicLengthFromCurvature","Mesh.CharacteristicLengthExtendFromBoundary",
):
    try: gmsh.option.setNumber(name, 0)
    except: pass
for name in ("Mesh.MeshSizeMin","Mesh.MeshSizeMax","Mesh.CharacteristicLengthMin","Mesh.CharacteristicLengthMax"):
    try: gmsh.option.setNumber(name, 1e-9 if "Min" in name else 1e12)
    except: pass

# Concentric spheres
v1  = gmsh.model.occ.addSphere(0, 0, 0, R1)
v2  = gmsh.model.occ.addSphere(0, 0, 0, R2)
v3  = gmsh.model.occ.addSphere(0, 0, 0, R3)

# Fragment -> one conforming mesh with shared interfaces at R1 and R2
gmsh.model.occ.fragment([(3, v3)], [(3, v1), (3, v2)])
gmsh.model.occ.synchronize()

# Physical groups: core (1), mid shell (2), outer shell (3)
vols = gmsh.model.getEntities(3)
def extent(tag):
    x0,y0,z0,x1,y1,z1 = gmsh.model.occ.getBoundingBox(3, tag)
    return max(abs(x0),abs(x1),abs(y0),abs(y1),abs(z0),abs(z1))
tags_sorted = [t for _, t in sorted(vols, key=lambda e: extent(e[1]))]
core_tag, mid_tag, shell_tag = tags_sorted

gmsh.model.addPhysicalGroup(3, [core_tag],  1); gmsh.model.setPhysicalName(3, 1, "core")
gmsh.model.addPhysicalGroup(3, [mid_tag],   2); gmsh.model.setPhysicalName(3, 2, "mid")
gmsh.model.addPhysicalGroup(3, [shell_tag], 3); gmsh.model.setPhysicalName(3, 3, "shell")

# (Optional) tag interfaces
surfs = gmsh.model.getEntities(2)
areas = [gmsh.model.occ.getMass(2, s[1]) for s in surfs]
iface_R1 = surfs[int(np.argmin([abs(a - 4*np.pi*R1*R1) for a in areas]))][1]
iface_R2 = surfs[int(np.argmin([abs(a - 4*np.pi*R2*R2) for a in areas]))][1]
gmsh.model.addPhysicalGroup(2, [iface_R1], 10); gmsh.model.setPhysicalName(2, 10, f"r={R1}")
gmsh.model.addPhysicalGroup(2, [iface_R2], 11); gmsh.model.setPhysicalName(2, 11, f"r={R2}")

# Smooth radial size field as ONE expression (no semicolons/assignments):
# w12 = 0.5*(1+tanh((r-R1)/W1)), w23 = 0.5*(1+tanh((r-R2)/W2))
# H(r) = H_CORE*(1-w12) + H_MID*w12*(1-w23) + H_SHELL*w23
expr = (
    f"{H_CORE}*(1 - 0.5*(1+tanh((sqrt(x*x+y*y+z*z)-{R1})/{W1}))) "
    f"+ {H_MID}*(0.5*(1+tanh((sqrt(x*x+y*y+z*z)-{R1})/{W1})))*(1 - 0.5*(1+tanh((sqrt(x*x+y*y+z*z)-{R2})/{W2}))) "
    f"+ {H_SHELL}*(0.5*(1+tanh((sqrt(x*x+y*y+z*z)-{R2})/{W2})))"
)
fid = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(fid, "F", expr)
gmsh.model.mesh.field.setAsBackgroundMesh(fid)

# Mesh + optional Netgen optimization
gmsh.model.mesh.generate(3)
try:
    gmsh.model.mesh.optimize("Netgen")
except:
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

gmsh.write("three_layer_sphere.msh")
gmsh.finalize()

# ---------- Convert & quick visualize ----------
mesh = meshio.read("three_layer_sphere.msh")
meshio.write("three_layer_sphere.vtu", mesh)
grid = pv.read("three_layer_sphere.vtu")

# Physical IDs -> flat array
region_ids = None
for key in ("gmsh:physical","physical"):
    if key in grid.cell_data:
        d = grid.cell_data[key]
        if isinstance(d, np.ndarray) and d.size == grid.n_cells: region_ids = d; break
        if isinstance(d, (list,tuple)):
            for a in d:
                if hasattr(a,"size") and a.size == grid.n_cells: region_ids = a; break
        if region_ids is not None: break
if region_ids is None:
    rcen = np.linalg.norm(grid.cell_centers().points, axis=1)
    region_ids = np.where(rcen <= R1+1e-8, 1, np.where(rcen <= R2+1e-8, 2, 3))
grid.cell_data["region_id"] = np.asarray(region_ids, dtype=np.int32)

# Clip to see inside
clip = grid.clip(normal=(1,0,0), origin=(0,0,0))
p = pv.Plotter()
p.add_mesh(grid.extract_surface(), color="lightgray", opacity=0.12)
p.add_mesh(clip, scalars="region_id", cmap="tab10", show_edges=True,
           edge_color="black", interpolate_before_map=False)
p.add_scalar_bar(title="region_id (1=core, 2=mid, 3=shell)")
p.show()
