# cubed_sphere_equal_cells.py
# pip install numpy pyvista
import numpy as np
import pyvista as pv

# ---------------- params ----------------
R_EARTH = 6371.0          # km
R_MIN   = 50.0            # inner cutoff to avoid singular center (km). Use small value, not 0.
TARGET_EDGE = 200.0       # ~desired tangential cell size near the surface (km)
ASPECT = 1.0              # target Δr / (r * Δα). 1.0 ~ "cube-ish" cells
WIRE = True               # overlay wireframe
R_ISO = 3000.0            # inner spherical surface to show (km); set None to skip

# ------------- equiangular cubed-sphere mapping -------------
def face_axes(face: str):
    if face == "+X": n,u,v = np.array([1,0,0.]), np.array([0,1,0.]), np.array([0,0,1.])
    elif face == "-X": n,u,v = np.array([-1,0,0.]), np.array([0,1,0.]), np.array([0,0,-1.])
    elif face == "+Y": n,u,v = np.array([0,1,0.]), np.array([0,0,1.]), np.array([1,0,0.])
    elif face == "-Y": n,u,v = np.array([0,-1,0.]), np.array([0,0,1.]), np.array([-1,0,0.])
    elif face == "+Z": n,u,v = np.array([0,0,1.]), np.array([1,0,0.]), np.array([0,1,0.])
    elif face == "-Z": n,u,v = np.array([0,0,-1.]), np.array([1,0,0.]), np.array([0,-1,0.])
    else: raise ValueError(face)
    return n,u,v

def cubed_sphere_patch(face: str, r_edges: np.ndarray, n_edge: int) -> pv.StructuredGrid:
    n,u,v = face_axes(face)
    a = np.linspace(-np.pi/4, np.pi/4, n_edge)   # face angles α
    b = np.linspace(-np.pi/4, np.pi/4, n_edge)   # face angles β
    TA, TB = np.tan(a), np.tan(b)
    TA, TB = np.meshgrid(TA, TB, indexing="ij")
    dirs = n[None,None,:] + TA[...,None]*u + TB[...,None]*v
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)     # unit sphere directions
    rr = r_edges[:,None,None,None] * np.ones_like(dirs)[None,...]
    pts = rr * dirs[None,...]                               # (nr, n_edge, n_edge, 3)
    X,Y,Z = pts[...,0], pts[...,1], pts[...,2]
    return pv.StructuredGrid(X,Y,Z)

# ------------- choose angular & radial resolution -------------
# Pick N_FACE from target surface edge length: Δα ≈ h / R, across face half-width π/4.
# Δα = (π/2)/(N_FACE-1)  ⇒  N_FACE ≈ 1 + (π/2)/Δα  with Δα ≈ TARGET_EDGE / R_EARTH
dalpha = TARGET_EDGE / R_EARTH
N_FACE = max(3, int(round(1 + (np.pi/2)/dalpha)))  # points per edge on each face
dalpha = (np.pi/2)/(N_FACE-1)                      # actual Δα used

# Choose number of radial points so that Δln r ≈ ASPECT * Δα  ⇒  N_R ≈ 1 + ln(R/r0)/(ASPECT*Δα)
N_R = max(3, int(round(1 + np.log(R_EARTH/R_MIN)/(ASPECT * dalpha))))
# Geometric radial edges: r_k = r0 * exp(k * Δln r), with Δln r = ln(R/r0)/(N_R-1)
r_edges = np.exp(np.linspace(np.log(R_MIN), np.log(R_EARTH), N_R))

print(f"N_FACE={N_FACE} (Δα={dalpha:.5f} rad, ~{np.degrees(dalpha):.2f}°), "
      f"N_R={N_R}, r0={R_MIN} km")

# ------------- build the 6 patches and combine -------------
faces = ["+X","-X","+Y","-Y","+Z","-Z"]
blocks = [cubed_sphere_patch(f, r_edges, N_FACE) for f in faces]
for b in blocks:
    b.point_data["radius"] = np.linalg.norm(b.points, axis=1)

ugrid = pv.MultiBlock(blocks).combine()
if "radius" not in ugrid.point_data:
    ugrid.point_data["radius"] = np.linalg.norm(ugrid.points, axis=1)

# ------------- visualize: outer clip + inner spherical surface -------------
clip = ugrid.clip(normal=(1,0,0), origin=(0,0,0))
surf_clip = clip.extract_surface().clean()

p = pv.Plotter()
p.add_mesh(
    clip, scalars="radius", cmap="viridis",
    show_edges=False, interpolate_before_map=False, opacity=0.95,
)
if WIRE:
    p.add_mesh(surf_clip, style="wireframe", color="black", line_width=1.0)

if R_ISO is not None:
    iso = ugrid.contour(isosurfaces=[R_ISO], scalars="radius")
    p.add_mesh(iso, color="tomato", opacity=0.85, show_edges=WIRE, edge_color="black")
    p.add_text(f"Inner isosurface r = {R_ISO:.0f} km", font_size=11)

# (optional) print rough aspect ratios at a few radii
for rprobe in [R_EARTH, 0.75*R_EARTH, 0.5*R_EARTH, 0.25*R_EARTH]:
    dr_over_r = np.log(R_EARTH/R_MIN)/(N_R-1)          # ≈ Δr/r
    tangential = rprobe * dalpha                        # ~ edge length along face
    radial = dr_over_r * rprobe                         # ~ Δr at that radius
    print(f"r~{rprobe:7.1f} km  tangential~{tangential:7.1f} km, "
          f"radial~{radial:7.1f} km,  ratio(rad/tan)~{radial/tangential:5.2f}")

p.add_scalar_bar(title="radius (km)")
p.show()
