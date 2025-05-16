import numpy as np
import cloudComPy as cc
import open3d as o3d
import time
import itertools
from sklearn.neighbors import NearestNeighbors
import laspy
import networkx as nx
import mistree_pp as mist
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from tqdm import tqdm
from stl import mesh

from os import listdir
from os.path import normpath, join


def list_all_descriptors(repository):

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


def list_all_centrelines(repository):

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


def list_all_plymeshes(repository):

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

    cloud = cc.ccPointCloud()
    cloud.coordsFromNPArray_copy(pointcloud)

    ref = cc.CloudSamplingTools.resampleCloudSpatially(cloud, minDistance= min_distance)
    (downsampled, _) = cloud.partialClone(ref)

    return downsampled.toNpArrayCopy()

# convenience function to downsample a numpy array spatially using the cloudcompy implementation 
def return_largest_component(pointcloud: np.ndarray, 
                             octree_level: int,
                             min_component_size: int)-> np.ndarray :
    """returns the largest connected component of a numpy.ndarray object (N x 3 matrix) of spatial coordinates using the dedicated CloudCompare
    algorithm. Returns a numpy.ndarray  (N x 3 matrix)"""

    cloud = cc.ccPointCloud()
    cloud.coordsFromNPArray_copy(pointcloud)

    out = cc.ExtractConnectedComponents([cloud], octree_level, min_component_size) 
    print(f"there were {len(out[1])} connected components.")
    if len(out[1]) > 0:
        largest = out[1][0]
    else: 
        largest = cloud

    # return the coordinates of the cloud 
    return largest.toNpArrayCopy()

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

## Mesh intersections 


def levi_civita_tensor(n):
    """returns the levi_civita tensor in n dimentions"""

    # initialise an empty multidimensional matrix (for n = 3, shape is (3,3,3))
    arr=np.zeros(tuple([n for _ in range(n)]))

    
    for x in itertools.permutations(tuple(range(n))):
        mat = np.zeros((n, n), dtype=np.int32)
        
        for i, j in zip(range(n), x):
            mat[i, j] = 1
        
        arr[x]=int(np.linalg.det(mat))
        
    return arr  


def return_ring(mesh, adjacent_triangles, P, nP, d, D):
    """Return a ring of vertices defining the inlet or outlet of a given section

    This requires mesh data, an adjacency matrix of the centroids, a plane defined by a point
    and its normal, and two threshold distances: out-of-plane distance, and within a radius of
    the initial point. 
    """

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

### legacy adjacency matrix. 
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
    
    # allow for different combinations of vertices by using np.roll on the cartesian coordinate triplet.
    combi_vectors = np.vstack([vectors, np.roll(vectors, 1, axis =1), np.roll(vectors, 2, axis =1)]).reshape((3,*vectors.shape))

    # initialise the indices of neighbouring triangles.
    adjacent_triangles = -999*np.ones((vectors.shape[0],3), dtype=int)
    
    for c, triangle_indices in enumerate(indices):
        combi_vec = combi_vectors[:,triangle_indices]
        orig_triangle = vectors[c]
    
        sharing_vertices = np.argwhere(np.alltrue(np.equal(orig_triangle,combi_vec), axis=3))[:,1]
        
        k = 0
        for i in np.unique(sharing_vertices):
            if np.argwhere(sharing_vertices==i).shape[0] == 2 and k < 3:
                #print(f"this index appears {i}, corresponding to triangle {triangle_indices[i]} appears")
                adjacent_triangles[c, k] = triangle_indices[i] 
                k+=1
    time_out = time.time()
    
    print(f"knn search complete and neighbouring triangles found in {time_out - time_in:.4f}s ({vectors.shape[0]} triangle neighbour sets.)")
    
    return adjacent_triangles

def adjust_plane(ring):
    P = np.mean(ring, axis=0)

    X = ring - P
    # covariance matrix
    Sigma =   X.T @ X
    # eigen value decomposition
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    nP= eigenvectors.T[np.argmin(eigenvalues)]

    # check that the distance to best fit plane is indeed correct (within tolerance). 
    #assert np.allclose(np.abs(X @ nP), 0, atol= 2e-1) 

    return P, nP

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
    n_clusters = len(np.unique(dbscan_labels))
    c = 1

    rings = {}
    nPi = []
    Pi = []

    while c <= n_clusters:
        cc0 = np.argmax(cc_sizes)    
        cc_sizes[cc0] = 0

        ring = edge_centroids[dbscan_labels == cc0]
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
        
        c+=1

    return rings
    
    # # select second largest
    # cc1 = np.argmax(cc_sizes)    
    # # find the two largest openings. 
    # ring1 = edge_centroids[dbscan_labels == cc0]
    # ring2 = edge_centroids[dbscan_labels == cc1]

    # print(f"found ring 2 with {len(ring2)} points.")

    # P1, nP1 = adjust_plane(ring1)
    # P2, nP2 = adjust_plane(ring2)

    # # reorient the normals towards the conduit. 
    # # check whether scalar product between normal solution and centre of mass of neighbouring points
    # # is positive or negative, and flip accordingly. 


    # for P, nP in zip(Pi), (nPi):
    #     # subsample nearby points.
    #     within_range = np.linalg.norm(mesh.centroids - P, axis =1) < 10
    #     within_plane = np.abs((mesh.centroids - P) @ nP) < 2

    #     neighbours = mesh.centroids[np.logical_and(within_plane, within_range)]
    #     com_neighbours = np.mean(neighbours, axis =0)

    #     # check scalar product sign, flip normal if necessary. 
    #     if (com_neighbours - P) @ nP < 0:
    #         nP *= -1


    # return dict(ring1=ring1, P1=P1, nP1=nP1, ring2=ring2,  P2=P2, nP2=nP2)
