#    Copyright (C) 2024-2026 by
#    Tanguy Racine <tanguy.racine@unine.ch>
#    All rights reserved.
#    MIT license.
#
"""
pc-centrelines geometries
=============

THe geometry module contains utilities.

"""

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.metrics import pairwise_distances
from shapely import Polygon, LineString, get_coordinates
from shapely.ops import split


def compute_ngon_perimeter(a: np.ndarray) -> float:
    """compute_ngon_perimeter calculates the perimeter of a closed polygon 
    whose vertex i coordinates are given by x[i] and y[i]

    :param a: a polygon vertex vector containing N coordinate pairs (N x 2 matrix)
    :type a: np.ndarray
    :return: the perimeter
    :rtype: float
    """

    return np.sqrt(np.sum((np.roll(a,1, axis = 0) - a)**2, axis = 1)).sum()


def compute_ngon_area(a: np.ndarray) -> float:
    """compute_ngon_area calculates the area of a closed polygon whose vertex i coordinates are given by x[i] and y[i]

    :param a: a polygon vertex vector containing N coordinate pairs (N x 2 matrix)
    :type a: np.ndarray
    :return: the area
    :rtype: float
    """

    x, y = a.T
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def compute_ngon_dh(a: float, p: float) -> float:
    """
    calculates the ahydraulic radius of a closed polygon whose vertex i coordinates are given by x[i] and y[i]
    

    ----------
    
    arguments:

        a -> scalar: area of a section
        p -> scalar: perimeter of a section 

    ----------
    
    returns :

        hydraulic diameter -> float : the area in input units squared 
    """
    
    return 4 * a / p


def compute_convex_hull(a: np.ndarray) -> np.ndarray:
    """
    calculates the convex hull of a closed polygon whose vertex i coordinates are given by x[i] and y[i]
    

    ----------
    
    arguments:

        a -> numpy.ndarray: area of a section
        p -> scalar: perimeter of a section 

    ----------
    
    returns :

        hydraulic diameter -> float : the area in input units squared 
    """
    hull = ConvexHull(a)

    hull_vertices = np.array((a[hull.vertices, 0], a[hull.vertices, 1])).T

    return hull_vertices


def make_ellipse(a: float, b: float, theta: float) -> np.ndarray:
    """
    calculate the cartesian coordinates of an ellipse centered on [0, 0]
    whose semi-axes are a and b respectively, rotated by angle theta
    

    ----------
    
    arguments:

        a -> scalar: area of a section
        b -> scalar: perimeter of a section
        theta -> scalar between [0, 2 pi] : angle of rotation

    ----------
    
    returns :

        hydraulic diameter -> float : the area in input units squared 
    """
    # define ellipse parametrically
    t = np.linspace(-np.pi, np.pi, 500)
    # calculate coordinates before rotation
    xy = np.vstack([a*np.cos(t), b*np.sin(t)]).T
    #rotation matrix
    rot = np.array([[np.cos(theta),-np.sin(theta)],
                      [np.sin(theta),np.cos(theta)]]).astype(np.float32)
    # return rotated ellipse. 
    return np.dot(xy, rot)



def centre_of_gravity(ngon: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    computes the centre of gravity of a 3d (but planar) n-gon

    ----------
    
    arguments:

        points -> np.array: a numpy array with N target coordinates (N x 3 matrix)
    ----------
    
    returns :

        cog -> np.array : a numpy array containing the coordinates of the centre of gravity (1 x 2 matrix)
    
    """
    n = len(ngon)
    # where ngon is the path determining the section by consecutive pairs of coordinates.
     
    denominator = np.zeros(n-2)
    numerator = np.zeros((n-2, 2))

    # test whether the 3dngon normal is colinear with any basic element. 

    test = np.any([np.allclose(col, 0, atol=1e-1) for col in ngon.T])

    if test:
        R = rotate(np.ones(3)/np.sqrt(3), np.pi / 4)
        Rprime = np.linalg.inv(R)

        ngon = (R @ ngon.T).T

    # determinant method (see: https://math.stackexchange.com/questions/90463/how-can-i-calculate-the-centroid-of-polygon)
    u = ngon[1:-1] - ngon[0]
    v = ngon[2:] - ngon[0]
    
    dets = u[:, 0]* v[:, 1] - u[:, 1]*v[:, 0]
    centroids = (ngon[0]+ngon[1:-1]+ngon[2:]) / 3
    numerator = np.sum(np.multiply(np.expand_dims(dets, -1), centroids), axis = 0) 
    denominator = np.sum(dets)

    cog = numerator / denominator

    if test:
        cog = Rprime @ cog

    if verbose == True:
        print("estimated area",  np.abs(np.sum(u[:, 0]* v[:, 1] - u[:, 1]*v[:, 0]))/ 2)

    return cog


def rotate(u, theta):
    """
    returns the rotation matrix given a unit vector u for direction and an angle theta.
    see also: orientations and rotations: https.//doi.org/10.1007/978-3-662-09156-2
    """
    
    costh = np.cos(theta)
    sinth = np.sin(theta)
    
    ux = u[0]
    uy = u[1]
    uz = u[2]
    
    R0 = [costh + ux**2 * (1 - costh), 
          ux * uy * (1 - costh) - uz * sinth, 
          ux * uz * (1 - costh) + ux * sinth]
    
    R1 = [uy * ux * (1 - costh) + uz * sinth,
          costh + uy**2 * (1 - costh), 
          uy * uz * (1 - costh) - ux * sinth]
    
    R2 = [uz * ux * (1 - costh) + uy * sinth, 
          uz * uy * (1 - costh) + ux * sinth, 
          costh + uz**2 * (1 - costh)]
    
    R = np.array([R0, R1, R2]).reshape((3,3))
    
    return R


def fit_ellipse(points2d, verbose = False, npoints = 500):

    # first part is PCA transform of the shape. 
    mean_points2d = np.mean(points2d, axis = 0)
    X = points2d - mean_points2d

    G = centre_of_gravity(X)
    area = compute_ngon_area(X)
    
    # covariance matrix
    Sigma = 1 / area *  X.T @ X
    # eigen value decomposition
    D, S = np.linalg.eig(Sigma)
    
    # find out the largest eigenvalue
    D_sorted = D[np.argsort(D)[::-1]]
    aspect_ratio = np.sqrt(D_sorted[0] / D_sorted[1])

    ellipse_rotation = np.rad2deg(np.arcsin(S[np.argsort(D)[::-1]][0, 1]))

    if verbose == True:
        print(G, area)
        print(D, S)
        print(aspect_ratio)
        
    D = np.diag(D)
    # k is a scaling factor to make the ellipse area equal to that of the ngon 
    k = np.sqrt(area  / (np.sqrt(np.linalg.det(Sigma))*np.pi))
    # determined ellipse
    t = np.linspace(0,2*np.pi, npoints)
    Y = np.vstack((np.cos(t), np.sin(t)))
    determined_ellipse = (k * S @ np.sqrt(D) @ Y).T + G + mean_points2d
    ellipse_area = compute_ngon_area(determined_ellipse)
    ellipse_perimeter = compute_ngon_perimeter(determined_ellipse)
    
    return determined_ellipse, ellipse_area, ellipse_perimeter, aspect_ratio, S, np.sqrt(D_sorted)*k, ellipse_rotation

def householder_transform(u: np.ndarray) -> np.ndarray:
    """Return the Householder transformation of vector u / it is a reflection about a hyperplan containin the origin
    """
    u = np.expand_dims(u, -1)
    return np.identity(u.shape[0]) - 2 *  (u @ u.T) / np.dot(u.T, u)

def vec2vec_rotation(f: np.ndarray, t: np.ndarray, verbose: bool = False) -> np.ndarray:

    """
    Rotates a vector f onto vector T by a single rotation along the axis of
    cross-product f x t.

    Returns rotation matrix R such that Rf = t

    ----
    Implementation of 'Möller, T., & Hughes, J. F. (1999). Efficiently Building a Matrix to Rotate One Vector to Another. 
    Journal of Graphics Tools, 4(4), 1–4. doi:10.1080/10867651.1999.10487509'
    """

    if verbose == True:
        print("troubleshooting:  ", t, f, np.linalg.norm(t), np.linalg.norm(f))

    t = t.astype(float) / np.linalg.norm(t)
    f = f.astype(float) / np.linalg.norm(f)

    # check that the vectors are not already near parallel. 
    if np.allclose(np.abs(np.dot(f, t)), 1, 0.01):

        if verbose == True:
            print("vectors seem to be near parallel, using reflections to compute rotation.")

        # product of two reflections is a rotation matrix, and easier to construct when 
        # vectors are parallel. 
        cond_x = (np.abs(f[0]) <  np.abs(f[1])) and (np.abs(f[0]) < np.abs(f[2]))
        cond_y = (np.abs(f[1]) <  np.abs(f[0])) and (np.abs(f[1]) < np.abs(f[2]))
        cond_z = (np.abs(f[2]) <  np.abs(f[0])) and (np.abs(f[2]) < np.abs(f[1]))

        if cond_x:
            p = np.array([1, 0, 0])
        elif cond_y:
            p = np.array([0, 1, 0])
        elif cond_z:
            p = np.array([0, 0, 1])
        else:
            p = np.array([0, 0, 0])

        
        # Compute reflections using Householder matrices
        A = householder_transform(p - f)
        B = householder_transform(p - t)

        R = B @ A
    
    
    else:
        v = np.cross(f, t)
        c = np.dot(f, t)
        h = (1 - c) / (np.dot(v, v))
    
        R = np.asarray([
        [c+h*v[0]**2, h*v[0]*v[1]-v[2], h*v[0]*v[2]+v[1]],
        [h*v[0]*v[1]+v[2], c+h*v[1]**2, h*v[1]*v[2]-v[0]],
        [h*v[0]*v[2]-v[1], h*v[1]*v[2]+v[0], c+h*v[2]**2]]
        )

    if np.allclose(t, R @ f, 0.001):
        if verbose == True:
            print("Rotation successful")
        return R
    else:
        if verbose == True:
            print("Rotation unsuccessful")
        return R
    
def to2D(points3d, nP, returnTransform=False, verbose = False):
    if np.sign(nP[2]) != 0:
        d = nP * np.sign(nP[2])
    else:
        d = nP
    
    d1 = d.copy()
    # project onto horizontal plane
    d1[2] = 0
    # define a normal vector
    d2 = np.array([0, 1, 0])
    # first rotation 
    R1 =  vec2vec_rotation(d, d1, verbose=verbose)
    pp = points3d - np.mean(points3d, axis=0)
    pp_vertical = R1 @ pp.T
    # second rotation
    R2 = vec2vec_rotation(d1, d2, verbose=verbose)
    pp_rotated = R2 @ pp_vertical
    # only assign the x and y direction for the 2d case
    points2d = pp_rotated.T[:, [0, 2]]
    
    if returnTransform == False:
        return points2d
    else:
        return points2d, R1, R2


def is_closed(a:np.ndarray, tol= 1e-1):
    """ 
    Is a 2d polygon section closed within a given tolerance?  
    """
    return np.linalg.norm(a[0]-a[-1]) < tol

def compute_max_pairwise_distance(a: np.ndarray)-> float:
    """ 
    Max || x1 - x2||  given x1 in a, x2 in a and x1 != x2
    """
    L = np.max(pairwise_distances(a))
    return L

def compute_rmax(a: np.ndarray, c:np.ndarray)-> float:
    """ 
    Max || x - c||  given x in a
    """

    rmax = np.max(np.linalg.norm(c-a, axis = 1))
    arg_rmax = np.argmax(np.linalg.norm(c-a, axis = 1))
    return rmax, arg_rmax

def compute_rmin(a: np.ndarray, c: np.ndarray)-> float:
    """ 
    Min || x - c||  given x in a
    """
    rmin = np.min(np.linalg.norm(c-a, axis = 1))
    arg_rmin = np.argmin(np.linalg.norm(c-a, axis = 1))
    return rmin, arg_rmin

def compute_rave(a: np.ndarray, c: np.ndarray)-> float:
    """ 
    Min || x - c||  given x in a
    """
    rave = np.mean(np.linalg.norm(c-a, axis = 1))
    return rave

def compute_circular_variance(a: np.ndarray, c : np.ndarray, A: float):

    spokes = np.linalg.norm(c-a, axis = 1)
    r_ave = np.mean(spokes)
    circular_variance = np.sum((spokes - r_ave)**2) / (A*r_ave**2)
    return circular_variance

def compute_polygon_solidity(a: np.ndarray):
    c_hull= ConvexHull(a)
    area_hull = compute_ngon_area(a[c_hull.vertices])
    area_polygon = compute_ngon_area(a)
    return area_polygon / area_hull


def return_below_fill_height(section, fill_height):
    # roll the section so any gap is at the top, always.
    polysection = Polygon(section)

    minx, miny = np.min(section, axis = 0)
    maxx, maxy = np.max(section, axis = 0)
    bounds = [minx, maxx, miny, maxy]
    
    line = LineString([[bounds[0], bounds[2]+fill_height*(bounds[3]-bounds[2])], [bounds[1],bounds[2]+fill_height*(bounds[3]-bounds[2])]])
    split_sections = split(polysection, line)

    filled_areas = []

    for split_section in split_sections.geoms:
        linear_ring = get_coordinates(split_section.exterior)

        if np.max(linear_ring[:, 1]) == bounds[2]+fill_height*(bounds[3]-bounds[2]):
            filled_areas.append(linear_ring)

    return filled_areas

def get_full_area(section, fill_height):
    filled_areas = return_below_fill_height(section, fill_height)
    full_area = 0
    for filled_area in filled_areas:
        full_area+= compute_ngon_area(filled_area)
    return full_area

def compute_ngon_wetted_perimeter(a: float, atol = 1e-3, keeptop=False) -> float:
    """
    calculates the ahydraulic radius of a closed polygon whose vertex i coordinates are given by x[i] and y[i]
    

    ----------
    
    arguments:

        a -> scalar: area of a section
        p -> scalar: perimeter of a section 

    ----------
    
    returns :

        wetted perimeter -> float : the area in input units squared 
    """
    
    # find near horizontal sections.
    y = a[:, 1]

    horizontal = (np.roll(y,1, axis = 0) - y)**2 < atol**2
    # are they near the top, if so remove them.
    near_ymax = (y - np.max(y))**2 < atol**2

    if keeptop == False:
        keep = ~np.logical_and(horizontal, near_ymax)
    else:
        keep = np.ones(len(a)).astype(bool)


    return np.sqrt(np.sum(((np.roll(a,1, axis = 0) - a)**2)[keep], axis = 1)).sum()


def compute_composite_dh(section, fill_fraction):
    filled_areas = return_below_fill_height(section, fill_fraction)

    wp = 0
    area = 0

    if fill_fraction == 1:
        assert len(filled_areas) == 1
        wp+= compute_ngon_perimeter(filled_areas[0])
        area += compute_ngon_area(filled_areas[0])
        
    else:
        for filled_area in filled_areas:
            wp += compute_ngon_wetted_perimeter(filled_area)
            area += compute_ngon_area(filled_area)



    if wp ==0:
        return 0
    else:
        return 2*area / wp
    
def describe(section: np.ndarray) -> dict:

    # call all predefined geometry functions
    cog = centre_of_gravity(np.vstack([section.T, np.zeros(len(section))]).T)[:-1]
    area = compute_ngon_area(section)
    perimeter = compute_ngon_perimeter(section)
    max_chord_length = compute_max_pairwise_distance(section)
    rmax, _ = compute_rmax(section, cog)
    rmin, _ = compute_rmin(section, cog)
    r_average = compute_rave(section, cog)
    solidity = compute_polygon_solidity(section)
    hydraulic_diameter = compute_ngon_dh(area, perimeter)
    circular_variance = compute_circular_variance(section, cog, area)
    closed = is_closed(section)

    bf_ellipse, ellipse_area, _, ellipse_aspect_ratio, ellipse_axes, ellipse_eigenvalues, ellipse_rotation = fit_ellipse(section)

    description = dict( 
    cog = cog,
    area = area,
    bf_ellipse = bf_ellipse,
    ellipse_area= ellipse_area,
    ellipse_aspect_ratio = ellipse_aspect_ratio,
    ellipse_axes = ellipse_axes,
    ellipse_eigenvalues = ellipse_eigenvalues,
    ellipse_rotation= np.deg2rad(ellipse_rotation), 
    perimeter = perimeter,
    solidity = solidity,
    hydraulic_diameter = hydraulic_diameter,
    max_chord_length = max_chord_length,
    rmax = rmax,
    rmin = rmin,
    r_average = r_average,
    circularity = rmin / rmax,
    gravelius_compactness = perimeter / (2* np.sqrt(np.pi*area)),
    circularity3 = (rmax - rmin) / rmax,
    deviation_from_inscribed_disk = np.pi * rmin**2 / area,
    deviation_from_circumscribed_disk = np.pi * max_chord_length**2 / (4*area),
    elongation = 4 / np.pi *  (area / r_average**2),
    isoperimetric_deficiency = 4* np.pi * area / perimeter**2,
    morton_spreading_index = 4 / np.pi * (area / max_chord_length**2),
    circular_variance = circular_variance,
    is_closed = closed)

    return description



DESCRIPTOR_MAP = {"area": {"name": "area", 
                           "unit": "m$^2$", 
                           "equation": "$card(x) \\rvert x \\in \\Omega$", 
                           "abbr": "A",
                           "bounds": [0, None]},
                "perimeter": {"name": "perimeter", 
                            "unit": "m", 
                            "equation": "$\\sum_{x \\in p}{V(x)}, \\text{V local weighting function}$", 
                            "abbr": "P",
                            "bounds": [0, None]},
                "max_chord_length": {"name": "max chord length", 
                            "unit": "m", 
                            "equation": "$max\\left\\Vert x_1 - x_2 \\right\\Vert x_1 \\in p, x_2 \\in p, x1 \\neq x_2$", 
                            "abbr": "L", "bounds": [0, None]},
                "rmin": {"name": "r min", 
                            "unit": "m", 
                            "equation": "$min\\left\\Vert x - c \\right\\Vert \\mid x \\in p$", 
                            "abbr": "$r_{min}$",
                            "bounds": [0, None]},
                "rmax": {"name": "r max", 
                            "unit": "m", 
                            "equation": "$max\\left\\Vert x - c \\right\\Vert \\mid x \\in p$", 
                            "abbr": "$r_{max}$",
                            "bounds": [0, None]},
                "r_average": {"name": "r average", 
                            "unit": "m", 
                            "equation": "$\frac{1}{card(p)} \\sum_{x \\in p}{\\left\\Vert x - c \\right\\Vert} $", 
                            "abbr": "$r_{ave}$",
                            "bounds": [0, None]},
                "hydraulic_diameter": {"name": "hydraulic diameter", 
                            "unit": "m", 
                             "equation": "$\\frac{4A}{P}$", 
                            "abbr": "$D_{H}$",
                            "bounds": [0, None]},
                "circularity": {"name": "circularity", 
                           "unit": "", 
                           "equation": "$\\frac{r_{min}}{r_{max}}$", 
                           "abbr": "$\\gamma_1$",
                           "bounds": [0, 1]},
                "circularity2": {"name": "circularity 2", 
                           "unit": "", 
                           "equation": "$\\frac{P^2}{A}$", 
                           "abbr": "$\\gamma_2$", 
                           "bounds": [4*np.pi, None]},
                "gravelius_compactness": {"name": "gravelius compactness coefficient", 
                           "unit": "", 
                           "equation": "$\\frac{P}{2 \\sqrt{\\pi A}}$", 
                           "abbr": "$\\gamma_c$", 
                           "bounds": [1,None]},
                "circularity3": {"name": "circularity 3", 
                           "unit": "", 
                           "equation": "$\\frac{r_{max} - r_{min}}{r_{max}}$", 
                           "abbr": "$\\gamma_3$",
                           "bounds": [0, None]},
                "deviation_from_inscribed_disk": {"name": "deviation from inscribed disk", 
                           "unit": "", 
                           "equation": "$\\frac{\\pi r_{min}^2}{A}$", 
                           "abbr": "$\\delta_{inscribed}$",
                           "bounds": [0, 1]},
                "deviation_from_circumscribed_disk": {"name": "deviation from inscribed disk", 
                           "unit": "", 
                           "equation": "$\\frac{\\pi r_{max}^2}{A}$", 
                           "abbr": "$\\delta_{circumscribed}$",
                           "bounds": [1, None]},
                 "elongation": {"name": "elongation", 
                            "unit": "", 
                            "equation": "$\\frac{4}{\\pi} \\frac{A}{r_{ave}^2}$", 
                            "abbr": "$E$", 
                            "bounds": [0, 4]},
                "isoperimetric_deficiency": {"name": "isoperimetric quotient", 
                            "unit": "", 
                            "equation": "$\\frac{4 \\pi A}{p^2}$", 
                            "abbr": "$d_{iso}$",
                            "bounds": [0, 1]},
                "morton_spreading_index": {"name": "Morton spreading index", 
                            "unit": "", 
                            "equation": "$\\frac{4}{\\pi} \\frac{A}{L^2}$", 
                            "abbr": "$\\gamma_{M}$",
                            "bounds": [0, 1]},
                "circular_variance": {"name": "circular variance", 
                            "unit": "", 
                            "equation": "$\\frac{\\sum_{x \\in \\Omega}{(\\left \\Vert x - c \\right \\Vert -r_{ave})^2}}{Ar_{ave}^2}$", 
                            "abbr": "$\\sigma^2_{circ}$", 
                            "bounds": [0, None]},
                "ellipse_aspect_ratio": {"name": "aspect ratio of ellipse axes", 
                            "unit": " ", 
                            "equation": " ", 
                            "abbr": "$R_e$",
                            "bounds": [1, None]},
                "ellipse_rotation": {"name": "ellipse inclination", 
                            "unit": "radians", 
                            "equation": " ", 
                            "abbr": "$\\theta_{e}$",
                            "bounds": [0, np.pi]},
                "solidity": {"name": "convex hull solidity", 
                            "unit": " ", 
                            "equation": "$card(x_1) / card(x_2) \\mid x_1 \\in \\Omega, x_2 \\in \\mathcal{C}(\\Omega)$", 
                            "abbr": "$S$",
                            "bounds": [0, 1]}
}