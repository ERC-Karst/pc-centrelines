import numpy as np
import cloudComPy as cc
import open3d as o3d
import time
import itertools
from sklearn.neighbors import NearestNeighbors
import laspy
import networkx as nx
import mistree as mist
from sklearn.neighbors import NearestNeighbors
from itertools import combinations

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
    
    R = [R0, R1, R2]
    
    return R

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
    cc0 = np.argmax(cc_sizes)    
    cc_sizes[cc0] = 0
    
    # select second largest
    cc1 = np.argmax(cc_sizes)    
    # find the two largest openings. 
    ring1 = edge_centroids[dbscan_labels == cc0]
    ring2 = edge_centroids[dbscan_labels == cc1]

    print(f"found ring 1 with {len(ring1)} points.")
    print(f"found ring 2 with {len(ring2)} points.")

    P1, nP1 = adjust_plane(ring1)
    P2, nP2 = adjust_plane(ring2)

    # reorient the normals towards the conduit. 
    # check whether scalar product between normal solution and centre of mass of neighbouring points
    # is positive or negative, and flip accordingly. 


    for P, nP in zip((P1,P2), (nP1, nP2)):
        # subsample nearby points.
        within_range = np.linalg.norm(mesh.centroids - P, axis =1) < 10
        within_plane = np.abs((mesh.centroids - P) @ nP) < 2

        neighbours = mesh.centroids[np.logical_and(within_plane, within_range)]
        com_neighbours = np.mean(neighbours, axis =0)

        # check scalar product sign, flip normal if necessary. 
        if (com_neighbours - P) @ nP < 0:
            nP *= -1


    return dict(ring1=ring1, P1=P1, nP1=nP1, ring2=ring2,  P2=P2, nP2=nP2)


def intersect(theMesh,adj_triangles, P, nP0):
    """
    Computes the intersection between a mesh and plane defined by a point and normal vector.
    Requires an adjacency matrix for touching triangles in the mesh to move from triangle
    to triangle during the section process. 

    Inputs: 
    -------
    
    Returns: 
    -------
    nP, M,M1M2, nOmega, W, area
    """
    
    # define the plane normal
    nP = nP0
    # build the levi_civita symbol
    eps = levi_civita_tensor(3)
    # extract the vertex position vectors.
    pos_vec = theMesh.vectors
    # extract the vertex normal vectors
    norm_vec = theMesh.normals

    # calculate the dot product of each triangle vertex to the plane of section: distance plane to vertex.
    PT_i_k = (P - pos_vec) @ nP

    # find all stricly positive or negative triplets.
    all_pos = np.all(np.equal(PT_i_k, np.abs(PT_i_k)), axis = 1)
    all_neg = np.all(np.equal(-PT_i_k, np.abs(PT_i_k)), axis = 1)

    # find all the triangles containing quantities PT_i_k of opposite signs 
    # i.e. exclude all wholly positive or negative triplets
    intersected_triangles = np.argwhere(~(all_neg | all_pos))
    
    # make a mutable list of as-yet unvisited triangles
    unvisited_triangles = [elem[0] for elem in intersected_triangles]
    
    # number of triangles to visit
    n = len(unvisited_triangles) 


    # find the triangle whose centroid coordinates are closest to the chosen point P. 
    Pcentroids = theMesh.centroids[intersected_triangles]-P
    
    try: 
        i =  np.argmin(np.linalg.norm(Pcentroids, axis = 2)) # starting index for the triangles to visit
    except ValueError:
        print("empty intersection")

        intersection = {"origin":P,
                    "plane_normal": nP,
                   "segment_middle_points": np.zeros((1, 3)),
                   "vertices":np.zeros((1,3)),
                   "vertex_normals": np.zeros((1,3)),
                   "rotation_axis": np.zeros(3),
                   "area":0}
        
        return intersection
    
    # counter for visited triangles
    triangle_counter = 0 

    # initialisation of segment middle points, facet normals, W and area.
    M = np.zeros((n, 3))
    # segment edges
    M1M2 = np.zeros((n, 3))
    PM1M2 = np.zeros((n+1,3))
    # facet normals
    nOmega = np.zeros((n, 3))
    # W
    W = np.zeros(3)
    # area
    area = 0
    # what is T0 ?? An initialised coordinate triplet
    T0 = np.zeros((3, 3))

    
    while triangle_counter < n: # stop when all triangles have been visited

        # for each triangle in the list of intersected triangles do the following: 

        # the index of the triangle relative to mesh indexation
        triangle_idx = intersected_triangles[i][0] 
        # find the distance to plane quantities for the vertex triplet at the ith triangle.
        PT_k = PT_i_k[triangle_idx]
        # select triplet of vertex position vectors
        T = pos_vec[triangle_idx]
        # select triplet of vertex normal vectors
        norm = norm_vec[triangle_idx]
    
        # label the summit alone in the half space with 0. # relabelling by rolling. Doing the same to the triangles. 
        if PT_k[1] * PT_k[2] < 0 and PT_k[0] * PT_k[1] < 0: 
            PT_k = np.roll(PT_k, 2)
            T = np.roll(T, 2, axis = 0)
        elif PT_k[1] * PT_k[2] < 0 and PT_k[0] * PT_k[1] > 0:
            PT_k = np.roll(PT_k, 1)
            T = np.roll(T,1, axis = 0)
        
        # the 0th label must be the solitary point, the other two must lie in the other half-space. 
        assert PT_k[1] * PT_k[2] > 0

        # find how far from the lone-summit the points of intersection are along segments
        alpha_i1 = PT_k[0] / (PT_k[0]-PT_k[1])
        alpha_i2 = PT_k[0] / (PT_k[0]-PT_k[2])

        # segment M_i1M_i2, lies in the place of intersection
        M_i1M_i2 = alpha_i2*(T[0] -T[2]) - alpha_i1*(T[0] -T[1])

        # intersection of the plane with segment T_i0T_i1
        M_i1 = T[0]- alpha_i1*(T[0] -T[1])
        # intersection of the plane with segment T_i0T_i2
        M_i2 = T[0]- alpha_i2*(T[0] -T[2])

        # coordinates of the middle point of segment M_i1M_i2
        M_i = M_i1-0.5 * M_i1M_i2

        # update the arrays
        M[triangle_counter] = M_i
        nOmega[triangle_counter] = norm

        # now perform some additional logic to find the correct next triangle to intersect.
        # check whether there is a flipped vector or not.
        if triangle_counter == 0:
            PM1M2[0] = M_i1
            PM1M2[1] = M_i2

        elif triangle_counter == 1:
           
            # check whether the previous point is the same as 1st or 2nd index.
            if np.allclose(M_i1, PM1M2[triangle_counter]) or np.allclose(M_i2, PM1M2[triangle_counter]):
                if np.allclose(M_i1, PM1M2[triangle_counter]):
                    PM1M2[triangle_counter+1] = M_i2
                else:
                    PM1M2[triangle_counter+1] = M_i1
                    
            elif np.allclose(M_i1, PM1M2[triangle_counter-1]) or np.allclose(M_i2, PM1M2[triangle_counter-1]):
                # we have to switch the first two.
                PM1M2[:2] = PM1M2.copy()[[1,0]]
                
                if np.allclose(M_i1, PM1M2[triangle_counter]):
                    PM1M2[triangle_counter+1] = M_i2
                else:
                    PM1M2[triangle_counter+1] = M_i1
            
            else:
                print("this might be invalid")
                if np.logical_not(np.all(np.equal(PM1M2[triangle_counter],M_i1))):
                    PM1M2[triangle_counter+1] = M_i1
                else:
                    PM1M2[triangle_counter+1] = M_i2
                    
        else:
            
            # check whether the previous point is the same as 1st index.
            if np.allclose(M_i1, PM1M2[triangle_counter]):
                PM1M2[triangle_counter+1] = M_i2
            
            elif np.allclose(M_i2, PM1M2[triangle_counter]):
                PM1M2[triangle_counter+1] = M_i1

        
            
        M1M2[triangle_counter] = PM1M2[triangle_counter+1] - PM1M2[triangle_counter]

        # update the area using the sum of determinants method
        area += np.linalg.det(np.array([M1M2[triangle_counter], M[triangle_counter]-P, nP])) / 2
        
        # update W vector. Cycle over the m elements of the resulting W vector.
        for m in range(3):
            num = 0
            Wi_m = []
            for j in range(3):
                for k in range(3):
                    # element wise multiplication of segment middle point jth element and the kth element of the triangle vector.
                    num+= eps[m,j,k]*(M[triangle_counter]-P)[j]*nOmega[triangle_counter,k]
                    
            denom = (M[triangle_counter] - P) @ nOmega[triangle_counter]
            
            
            det = np.linalg.det(np.array([M1M2[triangle_counter], M[triangle_counter]-P, nP]))
            
            # update W vector
            W[m] +=  num / denom * det 
        
        # find the neighbouring indices
        neighbour_indices = adj_triangles[triangle_idx]

        # update the list if unvisited triangles
        try:
            unvisited_triangles.remove(triangle_idx)
        except ValueError:
            print("neighbouring triangle not found")
            pass
            

        if neighbour_indices[0] in unvisited_triangles:
            triangle = neighbour_indices[0]
        elif neighbour_indices[1] in unvisited_triangles:
            triangle = neighbour_indices[1]
        elif neighbour_indices[2] in unvisited_triangles:
            triangle = neighbour_indices[2]
        else:

        # this will fail if a closed cycle section is found and there are remaining triangles.
            
            M = M[:triangle_counter+1].copy()
            M1M2 = M1M2[:triangle_counter+1].copy()
            PM1M2 = PM1M2[:triangle_counter+1].copy()
            nOmega = nOmega[:triangle_counter+1].copy()
            triangle_counter = n
                    
            
        # find the index of the next triangle to visit (in the list of intersected triangles)
        i = np.argwhere(intersected_triangles == triangle)[0][0] 
        
        # update the counter of visited triangles
        triangle_counter+=1 

    intersection = {"origin":P,
                    "plane_normal": nP,
                   "segment_middle_points": M,
                   "vertices":M1M2,
                   "vertex_normals": nOmega,
                   "rotation_axis": W,
                   "area":area}
    
    return intersection


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


def minimise_area(theMesh, adj_triangles, P, nP0, niter=20, eps=1e-1, step_adjust=1, global_shift = np.zeros(3)):
    s = time.time()
    # define the plane of section by a point and its normal. 
    nP = nP0

    i = 0
    # initialise some lists to keep track of the algorithm progress.
    track = []
    track.append(intersect(theMesh, adj_triangles,P+global_shift,nP))

    while i < niter:
        # get the intersection dictionary given a current plane normal nP
    
        W = track[-1]["rotation_axis"]
        A = track[-1]["area"]
        nP = track[-1]["plane_normal"]

        normW = np.linalg.norm(W)

        if np.linalg.norm(W) / np.abs(A) < eps:
            msg =f"normal vector already within desired tolerance of {eps}"
            return track, msg, 0. , 0

        else:
            # two rotations are possible around a unit vector colinear with W
            # amount of rotation is given by the norm of W
            R1 = rotate(W / normW,  - step_adjust * normW / np.abs(A)) 
            nP1 = R1 @ nP 
    
            R2 = rotate(W / normW, step_adjust * normW / np.abs(A)) 
            nP2 = R2 @ nP
    
            # update the normal.
            nPold = nP.copy()
    
            try:
                intersect_1 = intersect(theMesh, adj_triangles,P+global_shift,nP1)
            except (ValueError, AssertionError):
                intersect_1 = intersect(theMesh, adj_triangles,P+global_shift,nP)
                nP1 = nP
    
            try:
                intersect_2 = intersect(theMesh, adj_triangles,P+global_shift,nP2)
            except (ValueError, AssertionError):
                intersect_2 = intersect(theMesh, adj_triangles,P+global_shift,nP)
                nP2 = nP
    
            W1 = intersect_1["rotation_axis"]
            A1 = intersect_1["area"]
            W2 = intersect_2["rotation_axis"]
            A2 = intersect_2["area"]
    
            prog_crit_1 = np.abs(A1)
            prog_crit_2 = np.abs(A2)

            if prog_crit_1 < prog_crit_2:
                nP = nP1
            else:
                nP = nP2
                
            nP /= np.linalg.norm(nP)
            # find the best of two rotations for the last iteration
            best_crit = np.argmin([prog_crit_1,prog_crit_2])
            # set up two options to choose from
            options = [intersect_1, intersect_2]
            
            # append the best last iteration to minimisation tracking list
            if prog_crit_1 < eps or prog_crit_2 < eps:
                track.append(options[best_crit])
                i+=1
                msg = f"algorithm converged within desired tolerance of {eps} in {i} iterations"
                e = time.time()
                
                return track, msg, e-s, i
            else:
                track.append(options[best_crit])
                i+=1
                
                
        # update the minimisation iteration counter
        
        
    msg = f"algorithm failed to converge to the desired tolerance of {eps} in {niter} iterations"
    e = time.time()

    return track, msg, e-s, i

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
