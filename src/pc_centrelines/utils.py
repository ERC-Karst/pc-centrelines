#    Copyright (C) 2024-2026 by
#    Tanguy Racine <tanguy.racine@unine.ch>
#    All rights reserved.
#    MIT license.
#
"""
pc-centrelines utils
=============

This contains the general utilities. 

"""

import numpy as np
import open3d as o3d
import time
import itertools
from sklearn.neighbors import NearestNeighbors
import laspy
import networkx as nx
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from matplotlib.cm import tab20b
import mistree as mist

from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from tqdm import tqdm
from stl import mesh

from os import listdir
from os.path import normpath, join

import pc_centrelines.geometry as geom


def list_all_descriptors(repository : str) -> list:

    # list the different cave directories at the root directory level
    caves = [elem for elem in listdir(repository) if "." not in elem]

    # find unique passage filepaths
    passages_filepaths = []
    for cave in caves:
        # list all directory names within data
        new_passages = [elem for elem in listdir(join(repository, cave)) if "." not in elem]
        for new_passage in new_passages:
            passages_filepaths.append(normpath(join(repository, cave, new_passage, "geometric_descriptors", "section_descriptors.txt")))
    
    return passages_filepaths


def list_all_centrelines(repository : str) -> list:

    # list the different cave directories at the root directory level
    caves = [elem for elem in listdir(repository) if "." not in elem]

    # find unique passage filepaths
    passages_filepaths = []
    for cave in caves:
        # list all directory names within data
        new_passages = [elem for elem in listdir(join(repository, cave)) if "." not in elem]
        for new_passage in new_passages:
            passages_filepaths.append(normpath(join(repository, cave, new_passage, "geometric_descriptors", f"{cave}_{new_passage}_centreline.txt")))
    
    return passages_filepaths


def list_all_plymeshes(repository : str) -> list:

    # list the different cave directories at the root directory level
    caves = [elem for elem in listdir(repository) if "." not in elem]

    # find unique passage filepaths
    passages_filepaths = []
    for cave in caves:
        # list all directory names within data
        new_passages = [elem for elem in listdir(join(repository, cave)) if "." not in elem]
        for new_passage in new_passages:
            passages_filepaths.append(normpath(join(repository, cave, new_passage, "mesh", f"{cave}_{new_passage}_mesh_5cm.ply")))
    
    return passages_filepaths

# introspection and conversion to stl format.
def load_ply(filepath):
    import cloudComPy as cc

    meshply = cc.loadMesh(filepath)

    cloud = meshply.getAssociatedCloud()
    N = meshply.size()

    mesh_vectors = np.zeros((N, 3, 3))

    for c in tqdm(range(N)):
        triangle_indices = meshply.getTriangleVertIndexes(c)
        for vertex in range(3):
            mesh_vectors[c, vertex] = np.array(cloud.getPoint(triangle_indices[vertex]))

    meshstl = mesh.Mesh(np.zeros(len(mesh_vectors), dtype=mesh.Mesh.dtype))

    meshstl.vectors = mesh_vectors
    meshstl.update_centroids()
    meshstl.update_normals()
    return meshstl

def open_las(filepath: str) -> np.ndarray:
    with laspy.open(filepath) as file_header:
        las = file_header.read()

        coords = np.zeros((len(las.points), 3))

        for c, dim in enumerate("XYZ"):
            coords[:, c] = las.points[dim]* las.header.scales[c]+las.header.offset[c]
    return coords

# convenience function to downsample a numpy array spatially using the cloudcompy implementation 
def spatially_downsample(pointcloud: np.ndarray, min_distance: float)-> np.ndarray :
    """Downsamples a numpy.ndarray object (N x 3 matrix) of spatial coordinates using the dedicated CloudCompare
    algorithm. Returns a downsampled numpy.ndarray  (N x 3 matrix)"""
    import cloudComPy as cc

    cloud = cc.ccPointCloud()
    cloud.coordsFromNPArray_copy(pointcloud)

    ref = cc.CloudSamplingTools.resampleCloudSpatially(cloud, minDistance= min_distance)
    (downsampled, _) = cloud.partialClone(ref)

    return downsampled.toNpArrayCopy()

# convenience function to downsample a numpy array spatially using the cloudcompy implementation 
def return_largest_component(points: np.ndarray, 
                             eps: int,
                             min_component_size: int)-> np.ndarray :
    """returns the largest connected component of a numpy.ndarray object (N x 3 matrix) of spatial coordinates using the dedicated CloudCompare
    algorithm. Returns a numpy.ndarray  (N x 3 matrix)"""

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    min_points = min_component_size # Minimum number of points per cluster
    dbscan_labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    # find out the size of the connected components
    cc_sizes = [np.sum(dbscan_labels == c) for c in np.unique(dbscan_labels)  if c >=0]

    if cc_sizes:
        # select first largest
        n_clusters = len(np.unique(dbscan_labels)) - 1
        print(f"there are {n_clusters} clusters")
        
        cc0 = np.argmax(cc_sizes)    
        cc_sizes[cc0] = 0

        CC0 = points[dbscan_labels == cc0]
    else:
        CC0 = points

    print(len(points))
    return CC0

### legacy using cloudComPy    
    # import cloudComPy as cc

    # cloud = cc.ccPointCloud()
    # cloud.coordsFromNPArray_copy(pointcloud)

    # out = cc.ExtractConnectedComponents([cloud], octree_level, min_component_size) 
    # print(f"there were {len(out[1])} connected components.")
    # if len(out[1]) > 0:
    #     largest = out[1][0]
    # else: 
    #     largest = cloud

    # # return the coordinates of the cloud 
    # return largest.toNpArrayCopy()

# convenience function to go from numpy to o3d needed by LBC

def array_to_o3d(pointcloud: np.ndarray):
    """Converts a numpy.ndarray object (N x 3 matrix) with spatial coordinates to an open3d.geometry pointcloud
    object, as required for running the Laplacian-based contraction algorithm."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.cpu.pybind.utility.Vector3dVector(pointcloud)

    return cloud

# at each point in the centreline, use its two neighbours to estimate the passage direction. 
def return_centreline_parametrisation(centreline, dt=1):
    from scipy.interpolate import interp1d  

    curvilinear_coords = np.hstack([np.zeros(1), np.cumsum(np.linalg.norm(np.diff(centreline, axis=0), axis =1))])
    xinterp  = interp1d(curvilinear_coords, centreline[:, 0])
    yinterp = interp1d(curvilinear_coords, centreline[:, 1])
    zinterp = interp1d(curvilinear_coords, centreline[:, 2])

    tmax = np.floor(np.max(curvilinear_coords))
    ti = np.hstack([np.arange(0, tmax, dt), tmax])
    interpolated_centreline = np.vstack([xinterp(ti), yinterp(ti), zinterp(ti)]).T

    
    vectors = interpolated_centreline[:-2] - interpolated_centreline[2:]
    norm_vectors = np.linalg.norm(vectors, axis=1)
    normed_vectors = vectors / np.expand_dims(norm_vectors, -1)
    
    return interpolated_centreline, ti[1:-1], normed_vectors


# a convenience function to resample a centreline using a 1d interpolator. 
def parametrise_ct(ct : np.ndarray, eps: float)-> np.ndarray:
    """reparametrises a centreline using linear interpolation for upsampling and downsampling"""

    t = np.zeros(ct.shape[0])

    segments = np.linalg.norm(np.diff(ct, axis = 0), axis = 1)
    
    cumsum = np.cumsum(segments)

    L = np.max(cumsum)
    t[1:] =  cumsum / L

    x_interp  = interp1d(t, ct[:,  0])
    y_interp  = interp1d(t, ct[:,  1])
    if ct.shape[1] == 3:
        z_interp  = interp1d(t, ct[:,  2])

    # check if the interpolation is a downsampling or upsampling. 
    N  = len(ct) 
    if L/eps < N:
        # then we are downsampling. 
        ti = np.linspace(0,1, int(np.ceil(L/eps))) 

    else: 
        # now we work out the reparametrisation so that the average segment length is close to the required delta.
        ti = []
        t0 = 0
        for segment in segments:
            r = np.ceil(segment/eps)
            increments =  [t0+ i * segment / r for i in np.arange(0, r)]
            for increment in increments:
                ti.append(increment / L)
            t0 +=segment
        
        ti.append(1)
            
    if ct.shape[1] == 3:
        return np.vstack([x_interp(ti), y_interp(ti), z_interp(ti)]).T
    else:
        return np.vstack([x_interp(ti), y_interp(ti)]).T

def trim_adjust(ct1 : np.ndarray, ct2 : np.ndarray, eps : float) -> tuple:
    # we essentially reparametrise the lines to begin as close as possible. 

    ct1_ct2_nearest = np.argmin(np.linalg.norm(ct1[0] - ct2[[0, -1], :], axis =1))
    # this is only zero, or one. 
    flip = 1
    if ct1_ct2_nearest ==1:
        flip = -1

    ct1_parametrised = parametrise_ct(ct1, eps)
    ct2_parametrised = parametrise_ct(ct2[::flip], eps)

    # check whether a point in ct1 is closer to ct2_0. if so, we have to trim until there.
    idx_ct1_1 = np.argmin(np.linalg.norm(ct2[::flip][0] - ct1_parametrised,  axis=1))
    idx_ct2_1 = np.argmin(np.linalg.norm(ct1[0] - ct2_parametrised,  axis=1))

    idx_ct1_2 = np.argmin(np.linalg.norm(ct2[::flip][-1] - ct1_parametrised,  axis=1))
    idx_ct2_2 = np.argmin(np.linalg.norm(ct1[-1] - ct2_parametrised,  axis=1))

    P = ct1_parametrised[idx_ct1_1:idx_ct1_2]
    Q = ct2_parametrised[idx_ct2_1:idx_ct2_2]


    # projection of a point on a segment.
    # if qprime - Q[1] @ qprim - Q[0] = -1, then Q[0] = qprime
    # then, P[0] = pprime
    # we check now which way this should go. 
    Nq = (Q[1] - Q[0]) / np.linalg.norm(Q[1] - Q[0])
    pq = (P[0] - Q[0])
    Nqprime = np.dot(Nq, pq)*Nq
    qprime = Q[0]+ Nqprime
    
    if Nqprime @ (Q[1] - qprime) > 0:
        Q[0] = qprime
    else:
        Np = (P[1] - P[0]) / np.linalg.norm(P[1] - P[0])
        qp = (Q[0] - P[0])
        Npprime = np.dot(Np, qp)*Np
        pprime = P[0]+ Npprime
        P[0] = pprime

    Nq = (Q[-2] - Q[-1]) / np.linalg.norm(Q[-2] - Q[-1])
    pq = (P[-1] - Q[-1])
    Nqprime = np.dot(Nq, pq)*Nq
    qprime = Q[-1]+Nqprime
    if Nqprime @ (Q[-2] - qprime) > 0:
        Q[-1] = qprime
    else:
        Np = (P[-2] - P[-1]) / np.linalg.norm(P[-2] - P[-1])
        qp = (Q[-1] - P[-1])
        Npprime = np.dot(Np, qp)*Np
        pprime = P[-1]+ Npprime
        P[-1] = pprime

    return P, Q

## Mesh intersections 



def return_ring(mesh, adjacent_triangles, P, nP, d, D):
    """Return a ring of vertices defining the inlet or outlet of a given section

    This requires mesh data, an adjacency matrix of the centroids, a plane defined by a point
    and its normal, and two threshold distances: out-of-plane distance, and within a radius of
    the initial point. 
    """

    try:
        import mistree as mist
    except ModuleNotFoundError:
        import mistree_pp as mist

    # first find all triangles having two neighbours only (third column equal to an index of -999)
    edge_triangles = list(np.argwhere(adjacent_triangles[:,2] == -999).flatten())

    # return the coordinates of possible ring triangles centroids. 
    coordinates = mesh.centroids[edge_triangles]

    print("number of edge triangles :", len(coordinates))

    # dist to plane.
    dist_to_plane = (coordinates - P) @ nP

    # euclidean distance to point.
    dist_to_point = np.linalg.norm(coordinates - P, axis = 1)
    
    # perform the checks
    within_plane = np.abs(dist_to_plane) < d
    print("within the plane: ",np.sum(within_plane.astype(int)))

    within_range = dist_to_point < D
    print("within the range: ",np.sum(within_range.astype(int)))


    # find the ring triangles centroids, and return their coordinate triplets (unordered).
    ring = coordinates[np.logical_and(within_plane, within_range)]
    print("length of selected ring :", len(ring))

    # reorder the section by computing a minimum spanning tree and find the longest path traversing the tree.
    mst = mist.GetMST(x = ring[:, 0], y = ring[:, 1], z = ring[:, 2])

    degree,_,_,_, edge_index,_ = mst.get_stats(include_index=True, k_neighbours= 10)
    
    # build a graph using the edges.
    fullG = nx.graph.Graph(edge_index.T.tolist())

    # identify the leaves of the tree
    leaf = np.argwhere(degree==1).flatten().astype(int)

    print(f"this minimum spanning tree has {len(leaf)} leaves. Need to identify the best path.")

    # the shortest path linking our two extremities.
    best_shortest_path = 0
    best_comb = (0, 1)

    # iterate over combination of leaf indices to find the longest path (in terms of nodes) in the tree.
    for comb in combinations(list(range(len(leaf))), 2):
        i, j = comb
        centreline_path = nx.shortest_path_length(fullG, leaf[i], leaf[j])
        if centreline_path >= best_shortest_path:
            best_shortest_path = centreline_path
            best_comb = comb

    i, j = best_comb
    best_path = nx.shortest_path(fullG, leaf[i], leaf[j])
    
    return ring, best_path


def find_ring_order(unordered_ring):
    # reorder the section by computing a minimum spanning tree and find the longest path traversing the tree.
    mst = mist.GetMST(x = unordered_ring[:, 0], y = unordered_ring[:, 1], z = unordered_ring[:, 2])

    degree,_,_,_, edge_index,_ = mst.get_stats(include_index=True, k_neighbours= 10)
    
    # build a graph using the edges.
    fullG = nx.graph.Graph(edge_index.T.tolist())

    # identify the leaves of the tree
    leaf = np.argwhere(degree==1).flatten().astype(int)

    print(f"this minimum spanning tree has {len(leaf)} leaves. Need to identify the best path.")

    # the shortest path linking our two extremities.
    best_shortest_path = 0
    best_comb = (0, 1)

    # iterate over combination of leaf indices to find the longest path (in terms of nodes) in the tree.
    for comb in tqdm(combinations(list(range(len(leaf))), 2)):
        i, j = comb
        centreline_path = nx.shortest_path_length(fullG, leaf[i], leaf[j])
        if centreline_path >= best_shortest_path:
            best_shortest_path = centreline_path
            best_comb = comb

    i, j = best_comb
    best_path = nx.shortest_path(fullG, leaf[i], leaf[j])

    return np.array(best_path).astype(int)

### legacy adjacency matrix.
def find_adjacent_triangles(vectors, indices):
    # allow for different combinations of vertices by using np.roll on the cartesian coordinate triplet.
    combi_vectors = np.vstack([vectors, vectors[:, [2,0, 1]], vectors[:, [1, 2, 0]]]).reshape((3,*vectors.shape))

    # initialise the indices of neighbouring triangles.
    adjacent_triangles = -999*np.ones((vectors.shape[0],3), dtype=int)
    
    for c, triangle_indices in tqdm(enumerate(indices), total=len(indices)):
        combi_vec = combi_vectors[:,triangle_indices]
        orig_triangle = vectors[c]
    
        sharing_vertices = np.argwhere(np.all(np.equal(orig_triangle,combi_vec), axis=3))[:,1]
        
        k = 0
        for i in np.unique(sharing_vertices):
            if np.argwhere(sharing_vertices==i).shape[0] == 2 and k < 3:
                #print(f"this index appears {i}, corresponding to triangle {triangle_indices[i]} appears")
                adjacent_triangles[c, k] = triangle_indices[i] 
                k+=1
    
    return adjacent_triangles

def find_neighbours_with_knn(theMesh, n_neighbors=12):
    """
    extracts a n x 3 matrix of neighbouring triangle indices given a meshed surface containing n triangle.
    """
    time_in = time.time()
    
    # set up a nearest neighbour search.
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(theMesh.centroids)
    distances, indices = knn.kneighbors(theMesh.centroids)
    
    vectors = theMesh.vectors
    adjacent_triangles = find_adjacent_triangles(vectors, indices)
    time_out = time.time()
    
    print("knn search complete and neighbouring triangles found in:  \n")
    print(time_out - time_in, " s")
    
    return adjacent_triangles

def adjust_plane(ring : np.ndarray) -> tuple:
    P = np.mean(ring, axis=0)

    X = ring - P
    # covariance matrix
    Sigma =   X.T @ X
    # eigen value decomposition
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    nP= eigenvectors.T[np.argmin(eigenvalues)]

    return P, nP

def return_edge_midpoints(mesh, adjacent_triangles):
    edge_triangles = list(np.argwhere(adjacent_triangles[:,2] == -999).flatten())
    edge_centroids = mesh.centroids[edge_triangles]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(edge_centroids)
    eps = 0.5 # Distance threshold for points in a cluster
    min_points = 10 # Minimum number of points per cluster
    dbscan_labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    return edge_centroids, dbscan_labels

def return_source_target(mesh, adjacent_triangles, eps= 0.1) -> tuple:
    """Analyses the mesh geometry, finds the two largest ring openings, computes the centre of
    mass of each ring, adjusts a plane to the ring vertex coordinates, then computes a direction in which to shift
    the ring centre of mass to put the source into the mesh. 

    :param mesh: a mesh object_
    :type mesh: _stl.Mesh()_
    :param adjacent_triangles: a matrix containing the mesh triangle adjacency index matrix.
    :type adjacent_triangles: np.ndarray()
    :param eps: the voxel size by which to move the ring centre of mass inwards
    :type eps: float
    :return: a tuple of P1, nP1, P2, nP2 coordinates as np.ndarray()
    :rtype: tuple
    """
    ## find the rings (the largest two clusters of points lying on edges.)
    edge_midpoints, labels = return_edge_midpoints(mesh, adjacent_triangles)
    ## fit a plane into the direction of the passage. 
    # find the two largest connected components associated with rings. 

    # find out the size of the connected components
    cc_sizes = [np.sum(labels == c) for c in np.unique(labels)  if c >=0]
    # order the indices by decreasing size.
    cc_argsort = np.argsort(cc_sizes)[::-1]

    # index the rings by their dbscan label and return only the edge midpoint coordinates.
    ring1 = edge_midpoints[labels==cc_argsort[0]]
    ring2 = edge_midpoints[labels==cc_argsort[1]]

    # fit a plane to the cloud corresponding to edge midpoints, return a point in this plane and the 
    # normal vector.
    P1, nP1 = adjust_plane(ring1)
    P2, nP2 = adjust_plane(ring2)

    P1_neighbours = np.logical_and(np.abs((P1 - mesh.centroids) @ nP1)  < 2 *eps,
                               np.linalg.norm((P1 - mesh.centroids), axis=1)  < 10 *eps)
    P2_neighbours = np.logical_and(np.abs((P2 - mesh.centroids) @ nP2)  < 2 *eps, 
                               np.linalg.norm((P2 - mesh.centroids), axis=1)  < 10 *eps)

    sign_P1 = np.sign(nP1 @ (np.mean(mesh.centroids[P1_neighbours], axis =0) - P1))
    sign_P2 = np.sign(nP2 @ (np.mean(mesh.centroids[P2_neighbours], axis =0) - P2))

    P1_in = np.mean(ring1, axis =0)+ nP1*sign_P1*eps
    P2_in = np.mean(ring2, axis =0)+ nP2*sign_P2*eps 

    return P1_in, sign_P1*nP1, P2_in, sign_P2*nP2
                    

def return_rings(mesh, adjacent_triangles)-> dict:

    edge_triangles = list(np.argwhere(adjacent_triangles[:,2] == -999).flatten())
    edge_centroids = mesh.centroids[edge_triangles]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(edge_centroids)
    eps = 0.5 # Distance threshold for points in a cluster
    min_points = 10 # Minimum number of points per cluster
    dbscan_labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    # find out the size of the connected components
    cc_sizes = [np.sum(dbscan_labels == c) for c in np.unique(dbscan_labels)  if c >=0]

    # select first largest
    n_clusters = len(np.unique(dbscan_labels)) - 1
    print(f"there are {n_clusters} clusters")
    c = 1

    rings = {}

    while c <= n_clusters:
        cc0 = np.argmax(cc_sizes)    
        cc_sizes[cc0] = 0

        ring = edge_centroids[dbscan_labels == cc0]
        ring_mesh_indices = np.array(edge_triangles)[dbscan_labels == cc0]
        P, nP = adjust_plane(ring)
        
        #print(f"found ring  with {len(ring)} points.")

        within_range = np.linalg.norm(mesh.centroids - P, axis =1) < 10
        within_plane = np.abs((mesh.centroids - P) @ nP) < 2

        neighbours = mesh.centroids[np.logical_and(within_plane, within_range)]
        com_neighbours = np.mean(neighbours, axis =0)

        # check scalar product sign, flip normal if necessary. 
        if (com_neighbours - P) @ nP < 0:
            nP *= -1

        rings[f"ring{c}"] =  ring
        rings[f"P{c}"] =  P
        rings[f"nP{c}"] =  nP
        rings[f"ring{c}_mesh_indices"] = ring_mesh_indices
        
        c+=1

    return rings

def equivalent_dh(dhi, li):
    # harmonic mean along the section 
    return np.sum(li)**(1/4) * np.sum(li *dhi**(-4))**(-1/4)

## PLOTTERS 

def plotDH(sections,  idx, ax):
    dh = 4* geom.compute_ngon_area(sections[idx]) / geom.compute_ngon_perimeter(sections[idx])
    xi = np.linspace(0, 2*np.pi, 100)
    print(f"plotting the dh for section {idx} at {dh:.2f}m")
    # upper semi-circle
    ax.fill(dh/2*np.cos(xi), dh/2*np.sin(xi), alpha = 0.66, color = "grey", edgecolor=None)
    return ax

def plotCHull(sections,  idx, ax):
    chull = geom.compute_convex_hull(sections[idx])
    ax.fill(*sections[idx][:,[0, 1]].T, zorder = -9, color = "white")

    ax.fill(*chull[:,[0, 1]].T, zorder = -10, color = "sienna", alpha = 0.5)
    return ax


def plotR(sections, idx, ax, method):
    cog = geom.centre_of_gravity(sections[idx])

    ther, thearg = method(sections[idx], cog)

    ax.plot(*np.vstack([cog,sections[idx][thearg]]).T, color = "k", lw = 1)
    ax.scatter(*cog, marker="o", s=20, color="white", edgecolor="k")

def plotDescriptor(sections, descriptors, header, col, ax, 
                   ax_hist, 
                   ax_inset1, 
                   ax_inset2, 
                   inset_size = 10, 
                   last = False,
                   withmap = False,
                   with_autocovariance=False,
                   attr_n = 0, 
                   verbose = False):
    
    # coordinates for plotting in the insets
    descriptor = descriptors[:, col]
    xi = descriptors[:, -1]
    above_min_area = descriptors[:, 0] > 0.5
    is_closed = descriptors[:, -2]
    within_99 = np.logical_and(descriptor > np.nanquantile(descriptor, 0.005), descriptor < np.nanquantile(descriptor, 0.995))
    closed_within_99 = np.logical_and(is_closed.astype(bool), within_99)
    is_ok = np.logical_and(closed_within_99, above_min_area)


    descriptor_ok = descriptor[is_ok]
    xi_ok = xi[is_ok]

    sections_ok = np.array(sections, dtype=object)[is_ok]
    # indices to integers. 


    # indices of max and min descriptor value 
    argmax_descriptor = np.nanargmax(descriptor_ok)
    argmin_descriptor = np.nanargmin(descriptor_ok)



    _min = min(descriptor_ok)
    _max = max(descriptor_ok)
    _range = _max - _min

    if withmap==False:
    
        # Plotting along the curviline
        ax.plot(xi_ok, descriptor_ok, 
                "-o", 
                markersize = 3, 
                lw=0.75,
                color = tab20b(4*attr_n+1))

        ax.scatter(xi_ok[argmax_descriptor], descriptor_ok[argmax_descriptor], 
                color = "white",  
                edgecolor = tab20b(4*attr_n), 
                zorder = 100, 
                s = 20, 
                marker = "D")
        
        ax.scatter(xi_ok[argmin_descriptor], descriptor_ok[argmin_descriptor], 
                color = "white", 
                edgecolor = tab20b(4*attr_n+3), 
                zorder = 100, 
                s = 20, 
                marker = "D")
        
        ax.set_ylim(_min -0.1*_range, _max +0.1*_range)
    

    # Plotting the histogram
    nbins = int(1+ np.log2(len(descriptor_ok))) # sturges rule.
    ax_hist.hist(descriptor_ok, density = True, 
                 orientation = "horizontal", 
                 bins = nbins, 
                 color =tab20b(4*attr_n+3))

    # Gaussian KDE of the dataset, plotted on top of histogram. 
    yi = np.linspace(_min - 0.1*_range, _max + 0.1*_range, 100)
    kde = gaussian_kde(descriptor_ok).evaluate(yi)
    ax_hist.plot(kde, yi, lw = 3.5,c  = "white", zorder = 2)
    ax_hist.plot(kde, yi, lw = 1.5,c  = tab20b((4*attr_n)), zorder = 3)

    if with_autocovariance == False:
        # Plotting the insets
        ax_inset1.plot(*sections_ok[argmax_descriptor][:,[0, 1]].T,  color = tab20b(4*attr_n))

        ax_inset1.set_title(f"section {argmax_descriptor}")

        ax_inset2.plot(*sections_ok[argmin_descriptor][:,[0, 1]].T, color = tab20b((4*attr_n+3)))
        ax_inset2.set_title(f"section {argmin_descriptor}")


        # if hydraulic diameter. 
        if (header[col] == "hydraulic_diameter") and withmap == False: 
            plotDH(sections_ok, argmax_descriptor, ax_inset1)
            plotDH(sections_ok, argmin_descriptor, ax_inset2)

            li = np.diff(xi_ok)

            equiv_dh = equivalent_dh(descriptor_ok[:-1], li)
            ax.axhline(equiv_dh, ls="--", label = "equivalent hydraulic diameter", color = "k")

            ax.legend()
            
        elif header[col] == "solidity":
            plotCHull(sections_ok, argmax_descriptor, ax_inset1)
            plotCHull(sections_ok, argmin_descriptor, ax_inset2)

        elif header[col] == "rmin":
            plotR(sections_ok, argmax_descriptor, ax_inset1, geom.compute_rmin)
            plotR(sections_ok, argmin_descriptor, ax_inset2, geom.compute_rmin)
        
        elif header[col] == "rmax":
            plotR(sections_ok, argmax_descriptor, ax_inset1, geom.compute_rmax)
            plotR(sections_ok, argmin_descriptor, ax_inset2, geom.compute_rmax)
    
        # make cosmetic ajustments to insets 
        if with_autocovariance != True:
            for ax_inset in (ax_inset1, ax_inset2):
                ax_inset.set_aspect("equal")
                ax_inset.tick_params(top=True, right=True, left = False, bottom= False)
                ax_inset.xaxis.tick_top()
                ax_inset.yaxis.tick_right()
                ax_inset.yaxis.set_label_position("right")
                ax_inset.set_ylabel("z [m]")
                ax_inset.set_xlim(-inset_size, inset_size)
                ax_inset.set_ylim(-inset_size, inset_size)


                tick_pos = np.round(inset_size*2 /3)
                ax_inset.set_xticks([-tick_pos,0,tick_pos])
                ax_inset.set_yticks([-tick_pos,0,tick_pos])
    
    elif with_autocovariance== True:
        dt = xi[1] - xi[0]
        maxlag = int(2*len(descriptor)*dt / 3)
        print("max lag and dt: ", maxlag, dt)
        flags = np.logical_not(np.logical_and(within_99, above_min_area))

        #plot_autocorrelation_coefficients(ax_inset1, descriptor,maxlag, dt,colormap=tab20b, attrn= attr_n)
        
        plot_autocovariance(ax_inset1, ax_inset2, descriptor, flags, maxlag, dt, colormap=tab20b, attrn=attr_n, verbose=verbose)


    if withmap == False:
        ax.grid(lw = 0.5, ls="--")

        for spine in ["top","right"]:
            ax.spines[spine].set_visible(False)
            ax_hist.spines[spine].set_visible(False)
    else:
        for spine in ["top","right"]:
            ax_hist.spines[spine].set_visible(False)


    # annotate with some interesting trivia
    if last == True:
        ax.set_xlabel("Coordinate along centreline [m]")

def autocovariance(x, maxlag):
    autocov = np.zeros(maxlag)
    for lag in range(1, maxlag+1):
        covxlag = (x[lag:] - np.nanmean(x[lag:])) @ (x[:-lag] -np.nanmean(x[:-lag]))  / (len(x)-lag-1)
        autocov[lag-1] = covxlag
    return autocov

def autocorrelation_coefficients(x, maxlag):
    acf = np.zeros(maxlag)
    for lag in range(1, maxlag+1):
        covxlag = (x[lag:] - np.mean(x[lag:])) @ (x[:-lag] -np.mean(x[:-lag]))  / (len(x)-lag-1)
        lincorr = covxlag / (np.std(x[:-lag], ddof=1)*np.std(x[lag:], ddof=1))
        acf[lag-1] = lincorr
    return acf

def confidencebands(acf):
    n = len(acf)
    N = 2*n
    cb = np.zeros(n)
    # confidence bands
    cb[0] =  1 / np.sqrt(N)

    for lag in np.arange(2, n+1):
        SE_lag = np.sqrt((1+ 2*acf[:lag-1]@acf[:lag-1])/N)
        cb[lag-1] = SE_lag

    return cb

def analyse_autocovariance(signal, flags, maxlag, dt): 
    # assume the signal has nans. is evenly distributed. 
    x = np.arange(0,len(signal))*dt

    # fill in the signal with nans wherever the signal is to be ignored. 
    v_nan = signal.copy()
    x_nan = x.copy()
    x_nan[flags] = np.nan
    v_nan[flags] = np.nan

    # compute the autocovariance function at all lags from 0 to maxlag.
    n = len(v_nan)
    autocov_nan = np.zeros(maxlag)
    signal_centered = v_nan - np.nanmean(v_nan)

    for lag in range(maxlag):
        autocov_nan[lag] = np.nansum(signal_centered[:n-lag]*signal_centered[lag:]) / n

    # normalised autocovariance function 
    acf_nan = autocov_nan / autocov_nan[0]

    # find the integrated autocorrelation time using the method by Sokal (1997)
    j_max_v = np.arange(maxlag)
    tau_int_v  = np.zeros(maxlag)

    # compute the tau_int for many values of j_max (M, length of rectangular window)
    for j_max in j_max_v:
        tau_int_v[j_max] = 0.5 + np.sum(acf_nan[1: j_max+1])

    # find the largest integer j_max such that: j_max < C * tau_int_v[j_max]
    C = 5.0
    j_max = 0
    
    while j_max < C * tau_int_v[j_max]:
        if j_max == maxlag -1:
            break
        else:
            j_max+=1
        
    return x_nan, v_nan, autocov_nan, acf_nan, tau_int_v, j_max


def plot_autocovariance(ax1, ax2, signal, flags, maxlag, dt, colormap, attrn= 0):
    _, _, _, acf_nan, tau_int_v, j_max = analyse_autocovariance(signal, flags, maxlag, dt)

    ax1.semilogx(np.arange(1, maxlag)*dt, acf_nan[1:], 
                 label = f"$\\hat{{\\hat{{\\tau}}}}_{{int}}= {tau_int_v[j_max]*dt:.1f}$ m", 
                 color = colormap(4*attrn))

    ax1.set_xlabel("lag $\\mid t\\Delta t \\mid$ [m]")
    ax1.set_ylabel("$\\hat{\\hat{\\rho}}$(t)")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.axvline(tau_int_v[j_max]*dt, zorder = -10, ls = "--", color = "firebrick")

    C = 5
    ax2.semilogx(np.arange(1, maxlag)*dt, C*tau_int_v[1:])
    ax2.semilogx(np.arange(1, maxlag)*dt, range(1, maxlag), color = "k", ls ="--")
    ax2.set_ylim(0, 1.1*np.max(C*tau_int_v[1:]))
    ax2.set_ylabel("$c \\times \\hat{\\hat{\\tau_{{int}}}}$(M)")
    ax2.set_xlabel("window size M")


def plot_autocorrelation_coefficients(ax, x, maxlag, dt, colormap, attrn= 0):
    from scipy.stats import norm
    acf = autocorrelation_coefficients(x, maxlag)
    cb = confidencebands(acf)

    ax.plot(np.arange(1, maxlag+1)*dt, acf,"-o", color = colormap(attrn*4), markersize = 4)

    ax.set_xlabel("lag distance $j$ [m]")

    # confidence bands
    ax.plot(np.arange(1, maxlag+1)*dt,norm.ppf(0.025)*cb, color="k", ls = "--", lw = 0.75)
    ax.plot(np.arange(1, maxlag+1)*dt,norm.ppf(0.975)*cb, color="k", ls = "--", lw = 0.75)

    ax.scatter(np.arange(1, maxlag+1)*dt,acf+norm.ppf(0.025)*cb, color=colormap(4*attrn+2), s= 2)
    ax.scatter(np.arange(1, maxlag+1)*dt,acf+norm.ppf(0.975)*cb, color=colormap(4*attrn+2), s= 2)

    ax.set_ylabel("Autocorrelation function $\\hat{A}^{XX}_j$")

    ax.axhline(0, ls = "-", color = "k")

    return ax