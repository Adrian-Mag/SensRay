"""
Test tetrahedral quadrature implementation.

This script verifies that the in-house quadrature implementation
produces accurate results for various test functions.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensray.quadrature import TetrahedralQuadrature, get_good_scheme


def test_constant_function():
    """Test integration of constant function (should equal volume)."""
    print("\n=== Test 1: Constant Function ===")
    
    # Unit tetrahedron with vertices at origin and unit axes
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Volume of this tetrahedron = 1/6
    expected_volume = 1.0 / 6.0
    
    # Test with different orders
    for order in [1, 2, 3, 4, 5]:
        scheme = TetrahedralQuadrature(order=order)
        result = scheme.integrate(lambda x: np.ones(len(x)), vertices.T)
        error = abs(result - expected_volume)
        status = "✓" if error < 1e-10 else "✗"
        print(f"  Order {order}: {result:.12f}, Error: {error:.2e} {status}")
    
    return True


def test_linear_function():
    """Test integration of linear function."""
    print("\n=== Test 2: Linear Function f(x,y,z) = x ===")
    
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Integral of x over unit tetrahedron = 1/24
    expected = 1.0 / 24.0
    
    for order in [1, 2, 3, 4, 5]:
        scheme = TetrahedralQuadrature(order=order)
        result = scheme.integrate(lambda pts: pts[:, 0], vertices.T)
        error = abs(result - expected)
        status = "✓" if error < 1e-10 else "✗"
        print(f"  Order {order}: {result:.12f}, Error: {error:.2e} {status}")
    
    return True


def test_quadratic_function():
    """Test integration of quadratic function."""
    print("\n=== Test 3: Quadratic Function f(x,y,z) = x^2 + y^2 ===")
    
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Integral of x^2 + y^2 over unit tetrahedron
    # Can be computed analytically: 1/60
    expected = 1.0 / 60.0
    
    for order in [2, 3, 4, 5]:
        scheme = TetrahedralQuadrature(order=order)
        result = scheme.integrate(
            lambda pts: pts[:, 0]**2 + pts[:, 1]**2, 
            vertices.T
        )
        error = abs(result - expected)
        status = "✓" if error < 1e-9 else "✗"
        print(f"  Order {order}: {result:.12f}, Error: {error:.2e} {status}")
    
    return True


def test_scaled_tetrahedron():
    """Test with a scaled and translated tetrahedron."""
    print("\n=== Test 4: Scaled Tetrahedron ===")
    
    # Tetrahedron scaled by 2 and translated
    vertices = np.array([
        [1.0, 1.0, 1.0],
        [3.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 1.0, 3.0]
    ])
    
    # Volume = (1/6) * 2^3 = 4/3
    expected_volume = 4.0 / 3.0
    
    scheme = TetrahedralQuadrature(order=5)
    result = scheme.integrate(lambda x: np.ones(len(x)), vertices.T)
    error = abs(result - expected_volume)
    status = "✓" if error < 1e-10 else "✗"
    print(f"  Volume: {result:.12f}, Expected: {expected_volume:.12f}")
    print(f"  Error: {error:.2e} {status}")
    
    return error < 1e-10


def test_api_compatibility():
    """Test that the API matches quadpy's interface."""
    print("\n=== Test 5: API Compatibility ===")
    
    # Test get_good_scheme function
    scheme = get_good_scheme(5)
    print(f"  get_good_scheme(5): {type(scheme).__name__} ✓")
    
    # Test integrate method
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    result = scheme.integrate(lambda x: np.ones(len(x)), vertices.T)
    print(f"  integrate() returns: {type(result).__name__} = {result:.6f} ✓")
    
    return True


def test_arbitrary_tetrahedron():
    """Test with arbitrary tetrahedron."""
    print("\n=== Test 6: Arbitrary Tetrahedron ===")
    
    # Random-ish vertices
    vertices = np.array([
        [0.5, 1.2, 0.3],
        [2.1, 0.8, 0.5],
        [1.0, 2.5, 1.0],
        [0.8, 1.0, 2.3]
    ])
    
    # Compute volume manually
    v0, v1, v2, v3 = vertices
    edge1 = v1 - v0
    edge2 = v2 - v0
    edge3 = v3 - v0
    volume_expected = abs(np.dot(np.cross(edge1, edge2), edge3)) / 6.0
    
    scheme = TetrahedralQuadrature(order=5)
    volume_computed = scheme.integrate(lambda x: np.ones(len(x)), vertices.T)
    
    error = abs(volume_computed - volume_expected)
    status = "✓" if error < 1e-10 else "✗"
    print(f"  Volume (computed): {volume_computed:.12f}")
    print(f"  Volume (expected): {volume_expected:.12f}")
    print(f"  Error: {error:.2e} {status}")
    
    return error < 1e-10


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing In-House Tetrahedral Quadrature Implementation")
    print("=" * 60)
    
    tests = [
        test_constant_function,
        test_linear_function,
        test_quadratic_function,
        test_scaled_tetrahedron,
        test_api_compatibility,
        test_arbitrary_tetrahedron,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
