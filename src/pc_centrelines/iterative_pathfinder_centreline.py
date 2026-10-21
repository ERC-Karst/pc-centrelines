#    Copyright (C) 2024-2026 by
#    Tanguy Racine <tanguy.racine@unine.ch>
#    All rights reserved.
#    MIT license.
#
"""
pc-centrelines iterative pathfinder
=============

The iterative path finder module contains the iterative stepper centrelining algorithm

"""

import numpy as np
import networkx as nx
import open3d as o3d

from itertools import permutations
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import pairwise_distances
from pc_centrelines.geometry import rotate
from time import time
import tqdm as tqdm
import warnings

__all__ = ['iterative_pathfinder']

# analytical solution for a cylinder.


def section_area(A: float, n: np.ndarray) -> float:
    """Returns the section area of an oblique section taken across an elliptical cylinder in the x direction.
    The cylinder has an original section with area pi square units defined by the following equation:
    (y/A)^2 + z^2 = 1

    the angle of the section is given by n (the vector normal to te plane of section)
    :param A: _aspect ratio of the ellipse with unit area
    :type A: float
    :param n: vector normal to the plane of section
    :type n: np.ndarray
    :return: area
    :rtype: float
    """
    # we consider an elliptical cylinder in the z direction with an elliptical section defined by:
    # A = aspect ratio of the ellipse.
    # y**2 / A**2 + z**2 = 1
    nx, _, _ = n
    # nx = cos(theta)
    return np.pi * A / nx


def analytical_W(A: float, n: np.ndarray):
    x = np.array([1, 0, 0])
    return np.pi * A * np.cross(n, x) / n[0]**2

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

            with warnings.catch_warnings():
                warnings.filterwarnings("error")

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


def optimise_section(theMesh,
                     P: np.ndarray,
                     nP0=np.ndarray,
                     niter: int = 20,
                     eps: float = 0.1,
                     global_shift: np.ndarray = np.zeros(3),
                     verbose: bool = False) -> tuple:
    """A slightly modified fixed-step gradient descent algorithm for optimising a section orientation
    relative to the surrounding conduit. The idea is to find the orientation of a section with minimal area
    by rotation of the normal vector defining the plane of section, passing through P.

    :param theMesh: open3d mesh object
    :type theMesh: array.o3d
    :param P: position through which the section plane passes 
    :type P: np.ndarray
    :param nP0: vector defining the initial section plane
    :type nP0: np.ndarray
    :param niter: maximum number of iterations for the optimising loop, defaults to 20
    :type niter: int, optional
    :param eps: tolerance criterion for the end of the optimising loop, defaults to 0.1
    :type eps: float, optional
    :param global_shift: whether a shift needs to be applied to the mesh, defaults to np.zeros(3)
    :type global_shift: np.ndarray, optional
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :return: a log of the optimising algorithm, the time elapsed, a convergence message, the number of iterations needed.
    :rtype: tuple
    """

    s = time()
    # define the plane of section by a point and its normal.
    nP = nP0

    i = 0
    # initialise some lists to keep log of the algorithm progress.
    log = []
    try:
        log.append(mesh_intersect(theMesh, P+global_shift, nP))
    except ValueError:
        print("likely empty intersection")
        i = niter

    while i < niter:
        # get the intersection dictionary given a current plane normal nP
        if log[-1]:
            W = log[-1][3]
            A = log[-1][4]
            nP = log[-1][5]
            normW = np.linalg.norm(W)

        else:
            print("ending on a none type")
            msg = "ended on a none type intersection."
            e = time()
            return log, msg, e-s, i

        if verbose == True:
            print(A)
            print(f"||W || / |A| = {normW / np.abs(A):.3f}")
            print(f"{i}th iteration, normal_vector: ", np.round(nP, 2))

        if normW / np.abs(A) < eps:
            msg = f"normed W / A already within desired tolerance of {eps}"
            e = time()
            return log, msg, e-s, i

        else:
            if verbose is True:
                print("updating the normal.")
        # two rotations are possible around a unit vector colinear with W
            # amount of rotation is given by the norm of W

            R1 = rotate(W / normW, 0.5*normW / np.abs(A))

            R2 = rotate(W / normW, -0.5*normW / np.abs(A))

            intersect_1 = mesh_intersect(theMesh, P+global_shift, (R1 @ nP))
            intersect_2 = mesh_intersect(theMesh, P+global_shift, (R2 @ nP))

            try:
                nP1 = intersect_1[5]
                nP2 = intersect_2[5]
            except TypeError:
                msg = f"exited because of empty intersection."
                e = time()
                return log, msg, e-s, i

            A1 = intersect_1[4]
            A2 = intersect_2[4]
            W1 = intersect_1[3]
            W2 = intersect_2[3]

            prog_crit_1 = np.linalg.norm(W1)/np.abs(A1)
            prog_crit_2 = np.linalg.norm(W2)/np.abs(A2)

            if prog_crit_1 > normW / np.abs(A) and prog_crit_2 > normW / np.abs(A):
                if verbose is True:
                    print("neither rotation improves the W vector. ",
                          np.round(nP, 3))

                msg = f"algorithm stopped in {i} iterations"
                e = time()
                return log, msg, e-s, i

            else:
                if prog_crit_1 < prog_crit_2:

                    nP = nP1
                    normnP = np.linalg.norm(nP)
                    nP = nP / normnP
                    if verbose is True:
                        print("choosing the first option", np.round(nP, 3))

                else:
                    nP = nP2
                    normnP = np.linalg.norm(nP)
                    nP = nP / normnP
                    if verbose is True:
                        print("choosing the second option", np.round(nP, 3))

                # find the best of two rotations for the last iteration
                best_crit = np.argmin([prog_crit_1, prog_crit_2])
                # set up two options to choose from
                options = [intersect_1, intersect_2]

                # append the best last iteration to minimisation tracking list
                if prog_crit_1 < eps or prog_crit_2 < eps:
                    best_option = options[best_crit]
                    log.append(best_option)
                    i += 1
                    msg = f"algorithm converged within desired tolerance of {eps} in {i} iterations"
                    e = time()

                    return log, msg, e-s, i

                else:
                    log.append(options[best_crit])
                    i += 1

    msg = f"algorithm failed to converge to the desired tolerance of {eps} in {niter} iterations"
    e = time()

    return log, msg, e-s, i


def iterative_pathfinder(mesh_data,
                         P_start: np.ndarray,
                         P_end: np.ndarray,
                         nP0: np.ndarray,
                         L: float = 10,
                         ds: float = 0.5,
                         verbose: bool = False,
                         eps: float = 0.1) -> dict:
    """The cave path finder algorithm, starting at P_start and iteratively finding
    locally optimal section orientations, then stepping in this direction by the step ds.
    The algorithm stops under various conditions: 
        - a new point is outside the cave (failure)
        - a new point revisits a region of space already seen (failure)
        - a new point is close to the outlet plane, within a given tolerance (success)

    :param mesh_data: an open3d triangulated mesh object
    :type mesh_data: o3d.Mesh
    :param P_start: start point
    :type P_start: np.ndarray
    :param P_end: end point
    :type P_end: np.ndarray
    :param nP0: initial guess of the stepping direction
    :type nP0: np.ndarray
    :param L: the estimated length of the centreline, used as an extra stopping criterion, defaults to 10
    :type L: flow, optional
    :param ds: discretisation step, defaults to 0.5
    :type ds: float, optional
    :param verbose: whether to print statements about the progress, defaults to False
    :type verbose: bool, optional
    :param eps: tolerance as stopping criterion for the optimising loops, defaults to 0.1
    :type eps: float, optional
    :return: the centreline 
    :rtype: dict
    """

    # initialise the position along the conduit.
    P = P_start
    nP = nP0

    # initialise the lists to be updated during path finding.
    ct = []
    vertices = []
    areas = []
    normals = []
    n_it = []
    logs = []

    # append the first starting point.
    ct.append(P)
    # threshold distance to end point used as stopping criterion
    epsilon = ds / 2
    d = L
    # where 2 * L is the expected maximum curvilinear length of the centreline walk.
    max_it = np.floor(1.25 * L / ds)

    k = 0
    with tqdm.tqdm(total=max_it) as pbar:
        add_final = True
        _continue = True
        while _continue and k < max_it:
            # check that we are indeed farther away from the end point than at least one epsilon.
            assert d > epsilon

            try:
                log, _, _, iterations = optimise_section(
                    mesh_data, P, nP, verbose=verbose, eps=eps)

                best_section = log[-1][0]
                n_it.append(iterations)
                vertices.append(best_section)
                areas.append(log[-1][4])
                logs.append(log)

                # extract the reoriented normal
                nP_new = log[-1][5]

                # calculate the centre of gravity of the best polyline section found by area minimisation
                cog = np.mean(best_section, axis=0)
                if verbose is True:
                    print("centre of gravity of best section", cog)
                ct[-1] = cog
                # check we are not making a huge jump (no more than 3 times the current ds.)
                if len(ct) > 2:
                    fwd_step = np.linalg.norm(ct[-2] - ct[-1])
                    if fwd_step > 25*ds:
                        raise ValueError

                # check we are moving in the same half space as the one defined by the previous normal vector.
                step_sign = np.sign(nP @ nP_new)
                # step in this direction
                P = cog + ds*nP_new*step_sign
                # update the current normal vector value.
                nP = nP_new*step_sign

                if verbose is True:
                    print(
                        f"moving by step: {np.round(nP, 3)} to {np.round(P, 3)}.")

                # update the centreline and normal vector log
                ct.append(P)
                normals.append(nP)

                # the distance threshold should be checked by using
                # check the distance of end point in the plane...
                dplane = (P_end-P) @ nP

                if dplane < eps:
                    if d < 5*epsilon:
                        # stop if the end point is close to the plane of section
                        # and the barycentre is somewhat close to the end point.
                        _continue = False
                    else:
                        print(f"possibly too big a jump! {d:.5f}")

                d = np.linalg.norm(P_end-P)  # check the distance threshold

                # check also that we are not doubling back...

                if len(ct) >= 2:
                    dplane_previous = (ct[:-1]-P) @ nP
                    if np.any(dplane_previous < 0.25*ds):

                        d_previous = np.linalg.norm(ct[:-1]-P, axis=1)
                        if np.any(d_previous < 0.25*ds):
                            print("d_previous", d_previous)

                            _continue = False
                            print("seems we're doubling back to an known point.")

                if d < epsilon:
                    _continue = False

                k += 1  # update the iteration counter

                if verbose is True:
                    print(f"now moving to distance {d} from the end point.")

            except (ValueError, IndexError, TypeError):
                print(f"jump of {d:.4f}, for epsilon: {eps:.4f}")
                print("stopping process at empty intersection or too big a jump")
                k = max_it
                add_final = False
                pass
            pbar.update(1)

    if k != max_it and add_final:

        log, _, _,  iterations = optimise_section(
            mesh_data, P, nP, verbose=verbose, eps=eps)

        if log[-1] is not None:
            best_section = log[-1][0]
            n_it.append(iterations)
            vertices.append(best_section)
            areas.append(log[-1][4])
            logs.append(log)

    result = {"centreline": np.array(ct),
              "vertices": vertices,
              "areas": np.array(areas),
              "normals": np.array(normals),
              "n_iterations": n_it,
              "logs": logs}

    return result


def get_progress_logs(result) -> np.ndarray:
    """returns a progress log in the form of a 2D array
    for each row containing: 
         - the vector W, 
         - the area, 
         - the ratio thereof, 
         - the index of  kth section
         - the index of the jth iteration
         """
    logs = result["logs"]

    iteration_progress = []
    steps = []
    k = 0
    j = 0
    for log in logs:

        steps.append(k)
        for iter in log:

            iteration_progress.append((np.linalg.norm(iter[3]),
                                       iter[4],
                                       np.linalg.norm(iter[3]) / iter[4],
                                       k,
                                       j))
            j += 1
        k += 1

    return np.array(iteration_progress)
