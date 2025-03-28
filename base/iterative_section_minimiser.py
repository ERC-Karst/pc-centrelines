import numpy as np

# analytical solution for a cylinder. 

def section_area(A: float, n: np.ndarray) -> float:
    # we consider an elliptical cylinder in the z dirction with an elliptical section defined by:
    # A = aspect ratio of the ellipse.
    # x**2 / A**2 + y**2 = 1
    # nx, ny, nz = n
    # nz = cos(theta)
    return np.pi * A / n[2]

def analytical_W(A: float, n: np.ndarray):
    z = np.array(0,0,1)
    return np.pi * A * np.cross(n, z) / n[2]**2