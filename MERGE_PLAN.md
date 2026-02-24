# Branch `planetmodel_changes` → `main` — Audit & Merge Plan

Generated: 2026-02-24
Branch tip: `c35f634` (planetmodel_changes)
Main tip: `e038e16` (main)
Common ancestor: `1b4d826`

---

## 1. How the branches diverged

The student created `planetmodel_changes` from an old commit and worked on it
independently. A partial merge was made into `main` on 2026-02-02 (commit
`f07c894 "Upgrades to planet model data handling"`), which pulled some of the
student's layer-array refactor into main, but the two branches continued to
diverge after that. As a result:

- Both branches share a large common history but have **independent changes to
  the same files** — a plain `git merge` will produce conflicts.
- **`main` currently contains an unresolved merge-conflict marker** inside
  `sensray/planet_mesh.py` (`from_file`, around line 1940), committed by
  mistake. This must be fixed on `main` before the merge.
- The branch adds new public API (`cell_volumes`, `to_kernel_coefficients`,
  `normalize=` on `compute_sensitivity_kernel`) and new model files that do
  not exist on `main`.

---

## 2. File-by-file findings

### 2.1 `sensray/planet_model.py`

> **Verdict: keep the branch version; backport one main-only feature; clean up
> leftover artefacts.**

#### Internal data structure — the core change

| Aspect | `main` | `planetmodel_changes` | Decision |
|--------|--------|----------------------|----------|
| Layer storage | `self.layers[layer_name] = {depth: {vp,vs,rho}, ...}` — a dict keyed by depth float | `self.layers[layer_name] = {"depth": [], "vp": [], "vs": [], "rho": [], "radius": []}` — a dict of named NumPy arrays | **Keep branch.** Array-based storage is cleaner, faster (vectorised ops), and easier to serialise. |
| Initial layer seed | `self.layers = {}; current_layer_name = None` | `self.layers = {"surface": {...}}; current_layer_name = "surface"` | **Keep branch.** Avoids `None` guard throughout parsing. |
| Depth sorting | Hand-sorted with `sorted(layer_dict.items())` | `np.argsort` per layer | **Keep branch.** |

#### `_parse_nd_file` — artefacts to remove

The branch contains large commented-out blocks that are leftovers from the
old approach. They should be deleted entirely before the merge:

```python
# check if points trying to be added before any layer is defined
# if current_layer_name is None:
#     if len(self.layers) == 0:
#         current_layer_name = 'surface'
#     # else:
#     #     current_layer_name = 'unnamed_layer'
#     self.layers[current_layer_name] = {prop: [] for prop in props}
```

and

```python
# # Sort each layer's points by depth (ascending)
# for layer_name, layer_dict in self.layers.items():
#     sorted_items = sorted(layer_dict.items())
#     self.layers[layer_name] = dict(sorted_items)
```

#### `layerwise_linear_interp`

- **Branch**: exists as a named method; `get_property_at_depth` and
  `get_property_at_radius` delegate to it. The old `using_depths` parameter
  was removed this session (correct — it caused a discontinuity-ordering bug
  on the radius path).
- **Main**: the same logic is inlined inside `get_property_at_depth` without a
  named helper.
- **Decision**: keep the branch structure. The named method is useful and the
  bug fix must be preserved.

#### `get_property_profile` — **breaking API change**

| | `main` | `planetmodel_changes` |
|---|---|---|
| Single property return | `{'depth'/'radius': arr, 'value': arr}` | `{'depth'/'radius': arr, propname: arr}` |
| Multi property return | `{propname: {'depth'/'radius': arr, 'value': arr}, ...}` | `{'depth'/'radius': arr, 'vp': arr, 'vs': arr, ...}` |
| `average_discontinuities` param | Yes | **Removed** |

The branch return format is cleaner. The `average_discontinuities` parameter
should be assessed before the merge — it was a way to average the values at a
boundary rather than show the jump. If it is used nowhere outside `planet_model.py`
itself (currently only called from `plot_profiles`), it can be dropped.
If it is needed, it must be re-added to the branch.

**Current callers of `get_property_profile`:**

| Caller | Expects | Compatible with branch? |
|--------|---------|------------------------|
| `planet_model.py::plot_profiles` (line 587) | `profile[prop]` | ✅ branch already updated |
| `planet_mesh.py::populate_properties` (line ~340) | not called — uses `get_property_at_radius` instead | ✅ N/A |

#### `get_discontinuities` — **breaking API change**

| | `main` | `planetmodel_changes` |
|---|---|---|
| Return type | `List[float]` (depths or radii) | `Dict[str, Dict]` — richer structure |
| Return structure | `[3480.0, 1220.0, ...]` | `{'mantle': {'upper': {vp,vs,rho}, 'lower': {vp,vs,rho}, 'depth': float, 'radius': float}, ...}` |
| `as_depths` param | Yes | **Removed** |
| `include_radius` param | Yes | Yes |
| `outwards` param | Yes | Yes |

The richer dict return is a genuine improvement (callers can access property
jumps at the boundary). **All existing callers already use the dict form:**

| File | Line | Uses dict form? |
|------|------|-----------------|
| `demos/01_basic_usage.ipynb` | — | ✅ `.items()` |
| `demos/02_ray_tracing_kernels.ipynb` | — | ✅ `print()` only |
| `demos/03_projecting_models_on_mesh.ipynb` | — | ✅ `print()` only |
| `demos/05_lunar_P_PcP_kernels.ipynb` | — | ✅ `.items()` |
| `demos/06_mesh_discretization_comparison.ipynb` | — | ✅ `.items()` |
| `planet_mesh.py` (internal) | — | ✅ Not called |
| `planet_model.py::plot_profiles` | ~600 | ✅ branch already updated to dict form |

No callers use the old list form. Safe to keep the branch API.

#### `get_layer_info`

- Branch adds `outwards: bool = False` parameter. Safe addition with
  backward-compatible default.

#### `plot_profiles`

- Branch adds a formatted `prop_label` for the y-axis (e.g. `$v_{p}$ (km/s)`
  instead of the raw string `vp`). Keep this improvement.

#### `main`-only artefact to remove

`main` has a debug `print(self.radius)` statement inside `_parse_nd_file` that
was accidentally committed. It does not appear in the branch. This will
naturally disappear when the branch replaces the file.

---

### 2.2 `sensray/planet_mesh.py`

> **Verdict: the branch is a superset of main; keep branch; fix the conflict
> markers on main before merging.**

#### New methods on branch (not on main)

All were added this session and are clean additions:

| Addition | Description |
|----------|-------------|
| `cell_volumes` property | Returns `(n_cells,)` array of cell volumes in km³. For spherical: exact formula $V_j = \frac{4\pi}{3}(r_{j+1}^3-r_j^3)$. For tetrahedral: PyVista `compute_cell_sizes`. |
| `to_kernel_coefficients(K_tilde)` | Divides 1D or 2D kernel array by `cell_volumes`. Post-processing shortcut when G is already built. |
| `normalize=False` on `compute_sensitivity_kernel` | When `True`, divides by `cell_volumes` before returning. Works for all three call patterns. |
| Expanded Notes docstring on `compute_sensitivity_kernel` | Correct 3D derivation: $\tilde{K}_j = -L_j/v_j^2$, $K_j^\text{coeff} = \tilde{K}_j/V_j$. |

#### Unresolved conflict on `main` — **broken code, must fix before merge**

`main:sensray/planet_mesh.py` at lines 1940–1977 contains committed conflict
markers inside the `from_file` classmethod:

```
<<<<<<< HEAD
    # smart auto-detect: reads .npz vs .vtu, raises on ambiguity
    ...
=======
    mesh_type = metadata.get('mesh_type', 'spherical')
>>>>>>> 3db6bc7
```

The `HEAD` side (smart auto-detection) is clearly better. The branch correctly
removed both markers and the naive fallback line. The fix is: resolve the
conflict on `main` by keeping the full auto-detection block and deleting the
`=======` / `>>>>>>> 3db6bc7` lines and the one-liner between them.

---

### 2.3 `sensray/models/*.nd` — **branch is a superset**

| File | `main` | Branch |
|------|--------|--------|
| Standard Earth models (prem, ak135, etc.) | ✅ | ✅ |
| `M1.nd` (lunar) | ✅ | ✅ |
| `M2.nd` (lunar) | ❌ | ✅ |
| `M3.nd` (lunar) | ❌ | ✅ |
| `M2_resampled_on_M1.nd` | ✅ (main has updated version) | ✅ |
| `M3_resampled_on_M1.nd` | ✅ | ✅ |
| `weber_core.nd` | ✅ | ✅ |
| `weber_core_resampled_on_M1.nd` | ✅ | ✅ |
| `weber_core_smooth.nd` | ✅ | ✅ |
| `weber_core_smooth_resampled_on_M1.nd` | ✅ | ✅ |

The model files added on `main` after the split (`M2_resampled_on_M1.nd`,
`M3_resampled_on_M1.nd` updated versions) must be compared before merging to
ensure the branch has the right versions.

---

### 2.4 Demos

| File | `main` | Branch |
|------|--------|--------|
| `demos/01_basic_usage.ipynb` | ✅ latest | ✅ |
| `demos/02_ray_tracing_kernels.ipynb` | ✅ | ✅ |
| `demos/03_projecting_models_on_mesh.ipynb` | ✅ latest (Feb 24 update) | older |
| `demos/04_spherical_ray_tracing_kernels.ipynb` | ✅ | ✅ |
| `demos/05_lunar_P_PcP_kernels.ipynb` | ❌ | ✅ new |
| `demos/06_mesh_discretization_comparison.ipynb` | ❌ | ✅ new |

`demos/03` was updated on `main` on 2026-02-24 (`4acd751 "smarter mesh data
type detector"`). The branch has an older version. This needs to be checked —
the update may need to be carried forward to the branch before merging.

---

### 2.5 Deleted files — correct on branch, will clean main on merge

The branch correctly deletes all legacy scratch/development files:

- `develop/` directory (demo scripts, old notebooks)
- `test.py`, `test_octree.py`, `test_octree_2.py`, `test_planet_mesh_integration.py`
  (top-level test scripts that should have been in `tests/`)

These deletions are clean and will naturally propagate to main on merge.

---

### 2.6 Packaging

`pyproject.toml` and `setup.py` are identical on both branches after the
student's sync commit (`c35f634`). No action needed.

---

### 2.7 Tests

**There are no tests at present** — the `tests/` directory does not exist on
the branch (nor meaningfully on main). Before merging, a minimal test suite
should be created to prevent regressions.

---

## 3. Step-by-step merge plan

Work is ordered so that each step can be verified before the next begins.

---

### Step 1 — Fix `main`'s broken `planet_mesh.py` ✅ COMPLETE

**Commit**: `8383e90` on `main`
`fix: resolve committed conflict markers in from_file; remove debug print`

**What was done**:

1. `sensray/planet_mesh.py` — replaced the committed conflict markers in
   `from_file` (lines 1940–1977) with the smart auto-detection code (HEAD
   side). The naive one-liner `metadata.get('mesh_type', 'spherical')` that
   silently defaults to spherical was discarded in favour of the version that
   raises a clear `FileNotFoundError` or `FileExistsError` when the files on
   disk are ambiguous or missing.

2. `sensray/planet_model.py` — removed the accidental `print(self.radius)`
   debug statement that was printing the planet radius to stdout on every
   model load.

**Notes / future considerations**:
- The `planet_model.py` on `main` still uses the old dict-keyed layer
  structure (Step 2's branch cleanup is the last change before the full
  replacement arrives via the merge). Do not make further structural changes
  to `main:planet_model.py` — it will be overwritten wholesale by the merge.
- The smart auto-detect logic in `from_file` is now consistent between branches,
  which removes one source of conflict at merge time.

---

### Step 2 — Clean up `planet_model.py` on branch ✅ COMPLETE

**Commit**: `d7366a2` on `planetmodel_changes`
`cleanup: remove commented-out legacy parsing code; fix stale _parse_nd_file docstring`

**What was done**:

1. Removed the `# if current_layer_name is None:` commented-out block inside
   the `try` in `_parse_nd_file` — dead code from the old dict-keyed approach.
2. Removed the `# Sort each layer's points by depth` commented-out block —
   superseded by the `np.argsort` sort that follows immediately after.
3. Updated the `_parse_nd_file` docstring, which still described the old
   `{depth: {vp, vs, rho}}` structure; now documents the array-based layout
   with the full dict schema.

**Notes / future considerations**:
- The `_parse_nd_file` docstring comment `# sort by depth just in case` is
  technically redundant if .nd files always list depths in order, but it is
  cheap and defensive — leave it in.

---

### Step 3 — Decide on `average_discontinuities` ✅ COMPLETE (no action needed)

**Outcome**: `grep -rn "average_discontinuities" .` returned zero results across
all `.py` and `.ipynb` files in the repo. The parameter existed only on `main`
and was never called by anything. The branch correctly omits it — no backport
needed.

**Notes / future considerations**:
- If a user ever needs to average the property values at a discontinuity
  boundary (e.g. for a smooth model parameterisation), the feature can be
  added back cleanly given the new array-based layer structure. The logic
  would be: for each boundary, take the mean of `layer_above[prop][-1]` and
  `layer_below[prop][0]` and insert that averaged point in place of both.
  This is straightforward but not needed now.

---

### Step 4 — Check `demos/03` version *(on `planetmodel_changes`)* ✅ COMPLETE

**Outcome**: All 8 cell sources were identical between main and branch — the only
difference was that the branch had cleared outputs (`"outputs": []`) while main
had executed outputs. Took main's version to restore the executed state.

Commit `1462b34` on `planetmodel_changes`:
```
chore: restore executed outputs in demos/03 from main (sources were identical)
```

---

### Step 5 — Verify model files *(on `planetmodel_changes`)* ✅ COMPLETE (branch is correct)

**Outcome**: The branch's resampled `.nd` files are correct; main's are wrong.

- `M2_resampled_on_M1.nd`, `M3_resampled_on_M1.nd`: identical between branches.
- `weber_core_resampled_on_M1.nd`: **branch is correct**. At depth 0, source
  model has `Vp=1.0, Vs=0.5` — the branch matches exactly. Main has
  `Vp=1.66, Vs=0.89`, produced by the old buggy interpolation code.
- `weber_core_smooth_resampled_on_M1.nd`: **branch is correct**. The 738–1258 km
  layer of `weber_core_smooth.nd` is constant at `Vp=8.15, Vs=4.5`; the branch
  reflects this flatness, while main has spurious gradients from the old code.
- `weber_core.nd`: branch has a trailing blank line vs main — harmless.

**No action needed** — the branch files were regenerated after fixing the
interpolation bug and are physically correct. Do **not** take main's versions.

---

### Step 6 — Add a minimal test suite *(on `planetmodel_changes`)* ✅ COMPLETE

**Outcome**: 52 tests written and passing across two files:

- `tests/test_planet_model.py` — 34 tests covering loading, interpolation
  (`get_property_at_depth/radius`), `get_property_profile`, `get_discontinuities`
- `tests/test_planet_mesh.py` — 18 tests covering `SphericalPlanetMesh`,
  `cell_volumes` (exact values, additivity, error paths), `to_kernel_coefficients`
  (1D/2D shapes, correctness), and `populate_properties`

**Bonus fix**: `sensray/planet_model.py` on Python 3.8 raised `TypeError:
'type' object is not subscriptable` for the `dict[str, dict[str, float]]`
return annotation on `get_discontinuities`. Fixed by adding
`from __future__ import annotations` at the top of the file.

Commit `c14f516` on `planetmodel_changes`:
```
feat: add minimal test suite (52 tests); fix Python 3.8 dict[] annotation
```

Run with: `python -m pytest tests/ -v`

---

### Step 7 — Rebase/merge branch onto main *(final step)*

Once steps 1–6 are complete and all tests pass:

```bash
git checkout main
git merge planetmodel_changes --no-ff -m "merge: planetmodel_changes — array-based layers, kernel coefficients, lunar models"
```

Expected conflicts (because both branches touched these files after the common
ancestor):
- `sensray/planet_model.py` — main still has the old dict-keyed structure in
  places; resolve fully in favour of the branch
- `sensray/planet_mesh.py` — should be clean after Step 1 (branch is a
  strict superset of main's content)

If conflicts are large, an alternative is to use `git merge -s ours` for files
where the branch is definitively correct, followed by cherry-picking the
main-only commits that add value (the Feb 24 demo update).

---

## 4. Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| `get_property_profile` return format change breaks external users | Medium | All known callers already use branch format; documented clearly in changelog |
| `get_discontinuities` return format change | Low | All callers confirmed using dict form |
| Model file regression (resampled .nd files differ) | Medium | Step 5 diff check before merge |
| Missing `average_discontinuities` breaks something unknown | Low | Step 3 grep check |
| `main`'s broken conflict marker causes CI failure | High | Step 1 is prerequisite |
| No tests mean silent regressions | High | Step 6 creates baseline coverage |

---

## 5. Summary of what the merged `main` will gain

| Feature | Notes |
|---------|-------|
| Array-based layer storage in `PlanetModel` | Faster, cleaner, NumPy-native |
| `layerwise_linear_interp` as central dispatch | Fixes discontinuity-ordering bug on the radius interpolation path |
| `get_discontinuities` returns richer dict | Upper/lower property values at each boundary |
| `get_layer_info(outwards=)` parameter | Ordering control |
| Better `plot_profiles` axis labels | LaTeX-formatted |
| `cell_volumes` property on `PlanetMesh` | Exact 3D shell volumes for both mesh types |
| `to_kernel_coefficients()` method | Post-process G matrix without recomputing |
| `normalize=` on `compute_sensitivity_kernel` | Single-call path to $K_j^\text{coeff}$ |
| Correct 3D docstring for sensitivity kernels | $\tilde{K}_j/V_j$ derivation |
| `M2.nd`, `M3.nd` lunar models | New model files |
| `demos/05`, `demos/06` | Lunar kernel notebooks |
| Removal of `develop/`, `test*.py` clutter | Cleaner repo |
| Fixed conflict markers in `from_file` | Repo no longer broken |
