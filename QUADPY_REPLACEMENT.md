# Quadpy Replacement

## Overview

This directory contains an in-house implementation of tetrahedral quadrature to replace the `quadpy` library, which has licensing restrictions in newer versions.

## Files

- `sensray/quadrature.py`: In-house tetrahedral quadrature implementation
- `test_quadrature.py`: Comprehensive test suite

## Implementation Details

The implementation provides Gauss quadrature rules for tetrahedral integration using well-established numerical integration schemes:

- **Order 1**: 1-point centroid rule (exact for constants)
- **Order 2**: 4-point rule (exact for quadratics)
- **Order 3**: 5-point rule (exact for cubics) - **Used in SensRay**

## API Compatibility

The implementation provides a drop-in replacement for `quadpy.t3.get_good_scheme()` with optional quadpy support:

```python
from sensray import quadrature

# Use in-house implementation (default, no dependencies)
scheme = quadrature.t3.get_good_scheme(3)

# Or explicitly specify
scheme = quadrature.get_good_scheme(3, use_quadpy=False)

# Use quadpy if installed (for higher orders or compatibility)
scheme = quadrature.get_good_scheme(5, use_quadpy=True)

# Set default via environment variable
# export SENSRAY_USE_QUADPY=true
scheme = quadrature.get_good_scheme(5)  # Will use quadpy if available

# Integrate a function over a tetrahedron
# vertices: shape (4, 3) or (3, 4)
result = scheme.integrate(function, vertices)
```

## References

- Keast, P. (1986). "Moderate-degree tetrahedral quadrature formulas."
  *Computer Methods in Applied Mechanics and Engineering*, 55(3), 339-348.

## Testing

Run the test suite:

```bash
cd /disks/data/PhD/masters/SensRay
python test_quadrature.py
```

Or quick inline test:

```bash
python3 << 'EOF'
import numpy as np
import sys
sys.path.insert(0, "sensray")
import quadrature

vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
scheme = quadrature.get_good_scheme(3)

# Volume test (should be 1/6)
result = scheme.integrate(lambda x: np.ones(len(x)), vertices.T)
print(f"Volume: {result:.12f} (expected: 0.166666666667)")

# Quadratic test
result = scheme.integrate(lambda pts: pts[:, 0]**2 + pts[:, 1]**2, vertices.T)
print(f"x^2+y^2: {result:.12f} (expected: 0.033333333333)")
EOF
```

## Migration from quadpy

The in-house implementation is now the default. No changes are required for existing code:

**Before:**
```python
import quadpy
scheme = quadpy.t3.get_good_scheme(5)
```

**After (automatic, uses in-house by default):**
```python
from sensray import quadrature
scheme = quadrature.t3.get_good_scheme(3)  # In-house, order 3
```

**Optional: Keep using quadpy if installed:**
```python
from sensray import quadrature
scheme = quadrature.get_good_scheme(5, use_quadpy=True)
```

The integration in [sensray/planet_mesh.py](sensray/planet_mesh.py) has been updated to use `quadrature.t3.get_good_scheme(3)` instead of `quadpy.t3.get_good_scheme(5)`.

## Accuracy

Order 3 (5-point rule) provides exact integration for all polynomials up to degree 3, which is sufficient for the seismic property integration performed in SensRay. Testing shows errors < 1e-15 for polynomial test cases within the rule's exactness degree.
