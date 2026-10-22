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

import networkx as nx
import open3d as o3d
from itertools import permutations
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import pairwise_distances
from shapely import Polygon, LineString, get_coordinates
from shapely.ops import split
from typing import List
from tqdm import tqdm
from stl import mesh
from warnings import catch_warnings, filterwarnings

def compute_ngon_perimeter(a: np.ndarray) -> float:
    """compute_ngon_perimeter calculates the perimeter of a closed polygon 
    whose vertex i coordinates are given by x[i] and y[i]

    :param a: a polygon vertex vector containing N coordinate pairs (N x 2 matrix)
    :type a: np.ndarray
    :return: the perimeter
    :rtype: float
    """

    return np.sqrt(np.sum((np.roll(a, 1, axis=0) - a)**2, axis=1)).sum()


def compute_ngon_area(a: np.ndarray) -> float:
    """compute_ngon_area calculates the area of a closed polygon whose vertex i coordinates are given by x[i] and y[i]

    :param a: a polygon vertex vector containing N coordinate pairs (N x 2 matrix)
    :type a: np.ndarray
    :return: the area
    :rtype: float
    """

    x, y = a.T
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


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

def make_ellipse(a: float, b: float, theta: float, npoints=500) -> np.ndarray:
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
    t = np.linspace(-np.pi, np.pi, npoints)
    # calculate coordinates before rotation
    xy = np.vstack([a*np.cos(t), b*np.sin(t)]).T
    # rotation matrix
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]]).astype(np.float32)
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

    dets = u[:, 0] * v[:, 1] - u[:, 1]*v[:, 0]
    centroids = (ngon[0]+ngon[1:-1]+ngon[2:]) / 3
    numerator = np.sum(np.multiply(
        np.expand_dims(dets, -1), centroids), axis=0)
    denominator = np.sum(dets)

    cog = numerator / denominator

    if test:
        cog = Rprime @ cog

    if verbose is True:
        print("estimated area",  np.abs(
            np.sum(u[:, 0] * v[:, 1] - u[:, 1]*v[:, 0])) / 2)

    return cog


def rotate(u: np.ndarray, theta: float = 0) -> np.ndarray:
    """
    returns the rotation matrix given a unit vector u for direction and an angle theta.
    see also: orientations and rotations: https.//doi.org/10.1007/978-3-662-09156-2
    
    :param u: the direction vector
    :type u: np.ndarray
    :param theta: the angle of rotation in radians, defaults to 0
    :type theta: float, optional

    :return: a (3X3) rotation matrix
    :rtype: np.ndarray 
    """

    costh = np.cos(theta)
    sinth = np.sin(theta)

    u /= np.linalg.norm(u)

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

    R = np.array([R0, R1, R2]).reshape((3, 3))

    return R


def fit_ellipse(points2d : np.ndarray , verbose=False, npoints=500)-> tuple:
    """Determine a best-fit ellipse on a 2D closed planar curve

    :param points2d: the closed planar curve.
    :type points2d: np.ndarray
    :param verbose: whether to indicate some progress print statements, defaults to False
    :type verbose: bool, optional
    :param npoints: _description_, defaults to 500
    :type npoints: int, optional
    :return: a tuple contaning the parametrised best fit ellipse and its various components
    :rtype: type
    """


    # first part is PCA transform of the shape.
    mean_points2d = np.mean(points2d, axis=0)
    X = points2d - mean_points2d

    # find the centre of gravity
    G = centre_of_gravity(X)
    area = compute_ngon_area(X)

    # covariance matrix
    Sigma = 1 / area * X.T @ X
    # eigen value decomposition
    D, S = np.linalg.eig(Sigma)

    # find out the largest eigenvalue
    D_sorted = D[np.argsort(D)[::-1]]
    aspect_ratio = np.sqrt(D_sorted[0] / D_sorted[1])

    ellipse_rotation = np.rad2deg(np.arcsin(S[np.argsort(D)[::-1]][0, 1]))

    if verbose is True:
        print(G, area)
        print(D, S)
        print(aspect_ratio)

    D = np.diag(D)
    # k is a scaling factor to make the ellipse area equal to that of the ngon
    k = np.sqrt(area / (np.sqrt(np.linalg.det(Sigma))*np.pi))
    
    # determined ellipse using the following parametrisation
    t = np.linspace(0, 2*np.pi, npoints)
    Y = np.vstack((np.cos(t), np.sin(t)))
    determined_ellipse = (k * S @ np.sqrt(D) @ Y).T + G + mean_points2d
    ellipse_area = compute_ngon_area(determined_ellipse)
    ellipse_perimeter = compute_ngon_perimeter(determined_ellipse)

    return determined_ellipse, ellipse_area, ellipse_perimeter, aspect_ratio, S, np.sqrt(D_sorted)*k, ellipse_rotation


def householder_transform(u: np.ndarray) -> np.ndarray:
    """Return the Householder transformation of vector u / it is a reflection about a hyperplan containin the origin
    """
    u = np.expand_dims(u, -1)
    return np.identity(u.shape[0]) - 2 * (u @ u.T) / np.dot(u.T, u)


def vec2vec_rotation(f: np.ndarray, t: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Rotates a vector f onto vector T by a single rotation along the axis of
    cross-product f x t.

    Returns rotation matrix R such that Rf = t

    ----
    Implementation of 'Möller, T., & Hughes, J. F. (1999). Efficiently Building a Matrix to Rotate One Vector to Another. 
    Journal of Graphics Tools, 4(4), 1–4. doi:10.1080/10867651.1999.10487509'
    """

    if verbose is True:
        print("troubleshooting:  ", t, f, np.linalg.norm(t), np.linalg.norm(f))

    t = t.astype(float) / np.linalg.norm(t)
    f = f.astype(float) / np.linalg.norm(f)

    # check that the vectors are not already near parallel.
    if np.allclose(np.abs(np.dot(f, t)), 1, 0.01):

        if verbose is True:
            print(
                "vectors seem to be near parallel, using reflections to compute rotation.")

        # product of two reflections is a rotation matrix, and easier to construct when
        # vectors are parallel.
        cond_x = (np.abs(f[0]) < np.abs(f[1])) and (
            np.abs(f[0]) < np.abs(f[2]))
        cond_y = (np.abs(f[1]) < np.abs(f[0])) and (
            np.abs(f[1]) < np.abs(f[2]))
        cond_z = (np.abs(f[2]) < np.abs(f[0])) and (
            np.abs(f[2]) < np.abs(f[1]))

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
        if verbose is True:
            print("Rotation successful")
        return R
    else:
        if verbose is True:
            print("Rotation unsuccessful")
        return R


def to2D(points3d : np.ndarray, nP : np.ndarray, returnTransform : bool =False, verbose : bool =False)-> tuple:
    """Reproject a 3d closed planar into a 2D coordinate system in the plane. 
    First a rotation to verticalise the plane, then a rotation to align horizontal lines
    with the E-W cardinal directions is calculated.
    This preserves the vertical orientation of the planar curve: (the top remains as the top).

    :param points3d: the 3d closed planar curve
    :type points3d: np.ndarray
    :param nP: the pole to the plane
    :type nP: np.ndarray
    :param returnTransform: whether to output the transformation
    :type returnTransform: bool, optional
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :return: _description_
    :rtype: tuple
    """
    
    if np.sign(nP[2]) != 0:
        d = nP * np.sign(nP[2])
    else:
        d = nP

    d1 = d.copy()
    # project onto horizontal plane
    d1[2] = 0
    # define a normal vector
    d2 = np.array([0, 1, 0])
    # first rotation to verticalise the plane
    R1 = vec2vec_rotation(d, d1, verbose=verbose)
    pp = points3d - np.mean(points3d, axis=0)
    pp_vertical = R1 @ pp.T
    # second rotation to align the plane with the E-W cardinal direction.
    R2 = vec2vec_rotation(d1, d2, verbose=verbose)
    pp_rotated = R2 @ pp_vertical
    # only assign the x and y direction for the 2d case
    points2d = pp_rotated.T[:, [0, 2]]

    if returnTransform is False:
        R1, R2 =  None, None

    return points2d, R1, R2


def is_closed(a : np.ndarray, tol : float=1e-1) -> bool:
    """ 
    Is a 2d polygon section closed within a given tolerance?  
    """
    return np.linalg.norm(a[0]-a[-1]) < tol


def compute_max_pairwise_distance(a: np.ndarray) -> float:
    """ 
    Max || x1 - x2||  given x1 in a, x2 in a and x1 != x2
    """
    L = np.max(pairwise_distances(a))
    return L


def compute_rmax(a: np.ndarray, c: np.ndarray) -> float:
    """ 
    Max || x - c||  given x in a
    """

    rmax = np.max(np.linalg.norm(c-a, axis=1))
    arg_rmax = np.argmax(np.linalg.norm(c-a, axis=1))
    return rmax, arg_rmax


def compute_rmin(a: np.ndarray, c: np.ndarray) -> float:
    """ 
    Min || x - c||  given x in a
    """
    rmin = np.min(np.linalg.norm(c-a, axis=1))
    arg_rmin = np.argmin(np.linalg.norm(c-a, axis=1))
    return rmin, arg_rmin


def compute_rave(a: np.ndarray, c: np.ndarray) -> float:
    """ 
    Min || x - c||  given x in a
    """
    rave = np.mean(np.linalg.norm(c-a, axis=1))
    return rave


def compute_circular_variance(a: np.ndarray, c: np.ndarray, A: float):

    spokes = np.linalg.norm(c-a, axis=1)
    r_ave = np.mean(spokes)
    circular_variance = np.sum((spokes - r_ave)**2) / (A*r_ave**2)
    return circular_variance


def compute_polygon_solidity(point2d: np.ndarray)-> float:
    """Compute the ratio of section area to section convexhull area
    this is the convexhull solidity, varying from 0 to 1 (fully convex section)

    :param points2d: 
    :type points2d: np.ndarray
    :return: the actual solidity
    :rtype: float
    """

    c_hull = ConvexHull(point2d)
    area_hull = compute_ngon_area(point2d[c_hull.vertices])
    area_polygon = compute_ngon_area(point2d)

    return area_polygon / area_hull


def return_below_fill_height(points2d: np.ndarray, fill_height: float = 1) -> List[np.ndarray]:
    """return the section below a certain fill height, varying from 0 to 1.

    :param points2d: a 2d closed planar curve.
    :type points2d: np.ndarray
    :param fill_height: the ratio to the fill height to total passage height., defaults to 1
    :type fill_height: float, optional
    :return: _description_
    :rtype: 
    """

    # transform the closed planar curve to a shapely polygon instance.
    polysection = Polygon(points2d)
    # extract the bounds
    minx, miny = np.min(points2d, axis=0)
    maxx, maxy = np.max(points2d, axis=0)
    bounds = [minx, maxx, miny, maxy]

    # define the cut line
    line = LineString([[bounds[0], bounds[2]+fill_height*(bounds[3]-bounds[2])],
                      [bounds[1], bounds[2]+fill_height*(bounds[3]-bounds[2])]])
    
    # store the sections as a list of new polygons
    split_sections = split(polysection, line)

    filled_areas = []

    for split_section in split_sections.geoms:
        linear_ring = get_coordinates(split_section.exterior)
        # get the coordinates of each split section
        if np.max(linear_ring[:, 1]) == bounds[2]+fill_height*(bounds[3]-bounds[2]):
            # only append if the split area is actually below the fill height.
            filled_areas.append(linear_ring)
    # return the list of filled areas as a 2d np.ndarrays.
    return filled_areas


def get_full_area(section, fill_height):
    filled_areas = return_below_fill_height(section, fill_height)
    full_area = 0
    for filled_area in filled_areas:
        full_area += compute_ngon_area(filled_area)
    return full_area


def compute_ngon_wetted_perimeter(a: float, atol=1e-3, keeptop=False) -> float:
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

    horizontal = (np.roll(y, 1, axis=0) - y)**2 < atol**2
    # are they near the top, if so remove them.
    near_ymax = (y - np.max(y))**2 < atol**2

    if keeptop == False:
        keep = ~np.logical_and(horizontal, near_ymax)
    else:
        keep = np.ones(len(a)).astype(bool)

    return np.sqrt(np.sum(((np.roll(a, 1, axis=0) - a)**2)[keep], axis=1)).sum()


def compute_composite_dh(section, fill_fraction):
    filled_areas = return_below_fill_height(section, fill_fraction)

    wp = 0
    area = 0

    if fill_fraction == 1:
        assert len(filled_areas) == 1
        wp += compute_ngon_perimeter(filled_areas[0])
        area += compute_ngon_area(filled_areas[0])

    else:
        for filled_area in filled_areas:
            wp += compute_ngon_wetted_perimeter(filled_area)
            area += compute_ngon_area(filled_area)

    if wp == 0:
        return 0
    else:
        return 2*area / wp


def describe(section: np.ndarray) -> dict:

    # call all predefined geometry functions
    cog = centre_of_gravity(
        np.vstack([section.T, np.zeros(len(section))]).T)[:-1]
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

    bf_ellipse, ellipse_area, _, ellipse_aspect_ratio, ellipse_axes, ellipse_eigenvalues, ellipse_rotation = fit_ellipse(
        section)

    description = dict(
        cog=cog,
        area=area,
        bf_ellipse=bf_ellipse,
        ellipse_area=ellipse_area,
        ellipse_aspect_ratio=ellipse_aspect_ratio,
        ellipse_axes=ellipse_axes,
        ellipse_eigenvalues=ellipse_eigenvalues,
        ellipse_rotation=np.deg2rad(ellipse_rotation),
        perimeter=perimeter,
        solidity=solidity,
        hydraulic_diameter=hydraulic_diameter,
        max_chord_length=max_chord_length,
        rmax=rmax,
        rmin=rmin,
        r_average=r_average,
        circularity=rmin / rmax,
        gravelius_compactness=perimeter / (2 * np.sqrt(np.pi*area)),
        circularity3=(rmax - rmin) / rmax,
        deviation_from_inscribed_disk=np.pi * rmin**2 / area,
        deviation_from_circumscribed_disk=np.pi *
        max_chord_length**2 / (4*area),
        elongation=4 / np.pi * (area / r_average**2),
        isoperimetric_deficiency=4 * np.pi * area / perimeter**2,
        morton_spreading_index=4 / np.pi * (area / max_chord_length**2),
        circular_variance=circular_variance,
        is_closed=closed)

    return description


# Mesh intersections

def levi_civita_tensor(n):
    """returns the levi_civita tensor in n dimentions"""

    # initialise an empty multidimensional matrix (for n = 3, shape is (3,3,3))
    arr = np.zeros(tuple([n for _ in range(n)]))

    for x in permutations(tuple(range(n))):
        mat = np.zeros((n, n), dtype=np.int32)

        for i, j in zip(range(n), x):
            mat[i, j] = 1

        arr[x] = int(np.linalg.det(mat))

    return arr

def mesh_intersect(theMesh, P: np.ndarray, nP: np.ndarray, verbose=False) -> tuple:
    """Computes the intersection between a mesh and plane defined by a point and normal vector.
    Returns a tuple containing, among others: the normal vector used, the list of mesh vertices M, the list of segment midpoints M1M2,
    the list of intersected face normals (nOmega), the quantity W and the closed planar curve's area.

    :param mesh: An open3d mesh object
    :type mesh: _type_
    :param P: the point through which the intersection plane passes
    :type P: np.ndarray
    :param nP: the normal vector definining the intersection plane 
    :type nP: np.ndarray
    :param verbose: whether to print progress reports, defaults to False
    :type verbose: bool, optional
    :return: M_ordered, nOmega_ordered, M1M2_ordered, W, a
    :rtype: tuple
    """

    # build the levi_civita symbol
    eps = levi_civita_tensor(3)
    # extract the vertex position vectors.
    pos_vec = theMesh.vectors
    # extract the vertex normal vectors
    norm_vec = theMesh.normals

    # calculate the dot product of each triangle vertex to the plane of section: distance plane to vertex.
    PT_i_k = (P - pos_vec) @ nP

    # find all stricly positive or negative triplets.
    all_pos = np.all(np.equal(PT_i_k, np.abs(PT_i_k)), axis=1)
    all_neg = np.all(np.equal(-PT_i_k, np.abs(PT_i_k)), axis=1)

    # find all the triangles containing quantities PT_i_k of opposite signs
    # i.e. exclude all wholly positive or negative triplets
    intersected_triangles = np.argwhere(~(all_neg | all_pos)).flatten()

    # number of triangles to visit
    n = len(intersected_triangles)

    # this is a check to make sure the section actually crosses a minimum number of triangles.
    # also check that we are not outside the triangle.
    try:
        assert n > 3
    except AssertionError:
        print("empty intersection on too few triangles intersected.")
        return None

    # initialisation of segment middle points, facet normals, W and area.
    M = np.zeros((n, 3))
    # segment edges
    M1M2 = np.zeros((n, 2, 3))

    # facet normals
    nOmega = np.zeros((n, 3))
    # W
    W = np.zeros(3)
    # area
    area = 0

    PTc = np.linalg.norm(P - theMesh.centroids[intersected_triangles], axis=1)
    argminPTc = np.argmin(PTc)
    start_idx = intersected_triangles[argminPTc]

    if verbose == True:
        print("now choosing the first triangle whose centroid is closest to the plane central point.")
        print(start_idx, theMesh.centroids[start_idx], np.linalg.norm(
            P - theMesh.centroids[start_idx]))

    for c, i in enumerate(intersected_triangles):
        # find the distance to plane quantities for the vertex triplet at the ith triangle.
        PT_k = PT_i_k[i]

        # select triplet of vertex position vectors
        T = pos_vec[i]
        # select triplet of vertex normal vectors
        norm = norm_vec[i]

        # label the summit alone in the half space with 0. # relabelling by rolling. Doing the same to the triangles.
        if PT_k[1] * PT_k[2] < 0 and PT_k[0] * PT_k[1] < 0:
            PT_k = np.roll(PT_k, 2)
            T = np.roll(T, 2, axis=0)
        elif PT_k[1] * PT_k[2] < 0 and PT_k[0] * PT_k[1] > 0:
            PT_k = np.roll(PT_k, 1)
            T = np.roll(T, 1, axis=0)

        # the 0th label must be the solitary point, the other two must lie in the other half-space.

        try:
            assert PT_k[1] * PT_k[2] > 0
        except AssertionError:
            print("empty intersection on all triangle points in same half space.")
            mesh_intersect(theMesh, P, nP+np.random.normal(0,
                           0.005, 3), verbose=verbose)

        # find how far from the lone-summit the points of intersection are along segments
        alpha_i1 = PT_k[0] / (PT_k[0]-PT_k[1])
        alpha_i2 = PT_k[0] / (PT_k[0]-PT_k[2])

        # segment M_i1M_i2, lies in the place of intersection
        M_i1M_i2 = alpha_i2*(T[0] - T[2]) - alpha_i1*(T[0] - T[1])

        # check that it is true.

        try:
            assert np.abs(M_i1M_i2 @ nP) < 1e-2
        except AssertionError:
            print("empty intersection on points falling outside of tolerance.")
            mesh_intersect(theMesh, P, nP+np.random.normal(0,
                           0.005, 3), verbose=verbose)

        # intersection of the plane with segment T_i0T_i1
        M_i1 = T[0] - alpha_i1*(T[0] - T[1])
        # intersection of the plane with segment T_i0T_i2
        M_i2 = T[0] - alpha_i2*(T[0] - T[2])

        # coordinates of the middle point of segment M_i1M_i2
        M_i = M_i1-0.5 * M_i1M_i2

        # update the arrays
        M[c] = M_i
        M1M2[c] = [M_i1, M_i2]
        nOmega[c] = norm

    if len(M) == 0:
        raise ValueError

    M_old = M.copy()

    # select the cluster closest to point P.
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(M)
    epsilon = 0.2  # Distance threshold for points in a cluster
    min_points = 10  # Minimum number of points per cluster
    dbscan_labels = np.array(point_cloud.cluster_dbscan(
        eps=epsilon, min_points=min_points, print_progress=True))

    # find out the size of the connected components
    cc_sizes = [np.sum(dbscan_labels == c)
                for c in np.unique(dbscan_labels) if c >= 0]

    n_cc = len(cc_sizes)
    if verbose is True:
        print(f"there are {n_cc} connected components.")

    # check if there are strictly more than 1 connected components.
    if n_cc > 1:
        # select first largest
        c = 0

        rings = []

        while c <= n_cc:
            cc0 = np.argmax(cc_sizes)
            cc_sizes[cc0] = 0

            ring_bool = dbscan_labels == cc0
            ring_dic = {}
            ring_dic[f"ring{c}_bool"] = ring_bool

            ring_points = M[ring_bool]

            X = ring_points - np.mean(ring_points, axis=0)
            CovX = X.T @ X
            eigenvalues, _ = np.linalg.eig(CovX)

            ordered_eigenvalues = np.sort(eigenvalues)

            ring_dic[f"P_mean_ring{c}"] = np.linalg.norm(
                P - np.mean(M[ring_bool], axis=0))
            ring_dic[f"ring{c}_extent"] = np.prod(
                np.sqrt(ordered_eigenvalues[1:]))

            rings.append(ring_dic)

            c += 1

        if verbose is True:
            for c, ring in enumerate(rings):
                print("which rings should we use?", P,
                      ring[f"P_mean_ring{c}"], ring[f"ring{c}_extent"])

            print(np.argmin(
                [ring[f"P_mean_ring{c}"]/ring[f"ring{c}_extent"] for c, ring in enumerate(rings)]))

        thering = np.argmin(
            [ring[f"P_mean_ring{c}"] for c, ring in enumerate(rings)])

        theringbool = rings[thering][f"ring{thering}_bool"]
        M = M[theringbool].copy()
        M1M2 = M1M2[theringbool].copy()
        nOmega = nOmega[theringbool].copy()

    # beging the reordering of the nodes in the unstructured point set corresponding to the planar curve vertices
    pwdist = pairwise_distances(M)

    thetree = minimum_spanning_tree(pwdist)

    G = nx.Graph(thetree)

    # find the longest path in a tree using two successive bfs searches.
    bfs_1 = nx.bfs_tree(G, 0)
    first_endpoint = list(bfs_1.edges())[-1][-1]

    bfs_2 = nx.bfs_tree(G, first_endpoint)
    second_endpoint = list(bfs_2.edges())[-1][-1]

    best_path = nx.shortest_path(G, first_endpoint, second_endpoint)

    # segments need to be in order.
    M1M2_ord = np.zeros((n, 2, 3))

    for c, curr in enumerate(M1M2[best_path][:-1]):
        # check if the current should be flipped.
        next = M1M2[best_path][c+1]
        # only flip the current one, if the first index appears in the following segment.
        if np.allclose(next[1] - curr[0], np.zeros(3), atol=1e-5) or np.allclose(next[0] - curr[0], np.zeros(3), atol=1e-5):
            flipped = curr[::-1].copy()
            M1M2_ord[c] = flipped
        else:
            M1M2_ord[c] = curr.copy()

    M1M2_ord[-1] = M1M2[best_path][-1]

    nOmega_ord = nOmega[best_path]

    W = np.zeros((3, 1))

    for nOmega, (m1, m2) in zip(nOmega_ord, M1M2_ord):
        det = np.linalg.norm(np.cross(m2-m1, P-m1))
        detover2 = det / 2
        area += detover2
        PM = (m1+m2)/2-P

        # # update W vector. Cycle over the m elements of the resulting W vector.
        for m in range(3):
            num = 0
            for j in range(3):
                for k in range(3):
                    # element wise multiplication of segment middle point jth element and the kth element of the triangle vector.
                    num += eps[m, j, k]*PM[j]*nOmega[k]

            denom = PM @ nOmega

            with catch_warnings():
                filterwarnings("error")

                try:
                    if verbose == True:
                        print("num, denom, det", num, denom, det)
                        print("increment", num / denom * det)

                    inc = num / denom * det
                    W[m] += inc
                except Warning as e:
                    if verbose == True:
                        print("error found: ", e)
                    pass

    return M[best_path], nOmega_ord, M1M2_ord, W, area, nP


def describe_heuristics(themesh : mesh.Mesh, centreline : np.ndarray, tangent_vectors : np.ndarray, k =1):

    # read the centreline, start and end points and the tangent vectors.
    ct = centreline 
    P1 = ct[0]
    P2 = ct[-1]
    ct_vecs = tangent_vectors
    n = len(ct)
    thesections = np.arange(0, n)

    sections2d = []

    for thesection in tqdm(thesections[::k], total=int(n/k)):
        if (np.linalg.norm(ct_vecs[thesection]))**2 < 1e-6 :
            print("null vector!")
            
        else: 
            thevec = ct_vecs[thesection] / np.linalg.norm(ct_vecs[thesection])
            intersect = mesh_intersect(themesh, ct[thesection], thevec)
            if intersect is not None:
                polyline3d = intersect[0]
                section2d, _, _ = to2D(polyline3d, ct_vecs[thesection], returnTransform=True, verbose=False)
                sections2d.append(section2d)

    # the three descriptors under consideration are these. 
    
    areas = []
    aspect_ratios = []
    dhs = []

    tortuosity = np.sum(np.linalg.norm(ct[1:]-ct[:-1], axis =1)) / np.linalg.norm(P2-P1)


    for section2d in sections2d:
        try:
            description = describe(section2d)
            areas.append(description["area"])
            dhs.append(description["hydraulic_diameter"])
            aspect_ratios.append(description["ellipse_aspect_ratio"])
            
        except ValueError:
            print("invalid section")   
    
    return areas, np.mean(areas), aspect_ratios, np.mean(aspect_ratios), dhs, np.mean(dhs), tortuosity

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
                                            "bounds": [1, None]},
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
