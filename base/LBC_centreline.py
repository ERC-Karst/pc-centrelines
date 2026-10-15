import numpy as np

try:
    import mistree as mist
except ModuleNotFoundError:
    import mistree_pp as mist
    
import networkx as nx

from itertools import combinations
from time import time
from utils import array_to_o3d, return_largest_component

from laplacian import LBC

__all__ = ['return_LBC_centreline']


CT_ARGS = dict(keep_adjacent_triangles=True, centreline_down_sample= 0.5)
LBC_ARGS = dict(init_attraction = 0.5, init_contraction = 0.5, down_sample = 0.5)

def return_LBC_centreline(meshdata, 
                          ct_args: dict = CT_ARGS, 
                          lbc_args: dict = LBC_ARGS) -> dict:
    
    """A function encapsulating Laplacian-Based-contraction on a mesh

    :param mesh: An open3d mesh object
    :type mesh: _type_
    :param adjacent_triangles: a numpy array.
    :type adjacent_triangles: _type_
    :param ct_down_sample: _description_, defaults to CENTRELINE_DOWN_SAMPLE
    :type ct_down_sample: float, optional
    :param lbc_args: _description_, defaults to LBC_ARGS
    :type lbc_args: dictionary, optional
    :return: An array of 3d coordinates
    :rtype: np.ndarray
    """

    pcd = meshdata.centroids
    s = time()
    print(f"read file in {time() - s:.4f}s")
    
    s = time()

    # local shift
    local_shift = np.mean(pcd, axis = 0)

    # convert to open3d object-
    shifted_pcd = array_to_o3d(pcd - local_shift)

    # set up the Laplacian-based contraction algorithm 
    lbc = LBC(point_cloud=shifted_pcd, **lbc_args)
    print(f"contracted file in {time() - s:.4f}s")
    s = time()
    # extract the skeleton (this step takes a little while) 
    lbc.extract_skeleton()

    print(f"extracted skeleton in {time() - s:.4f}s")
    s = time()

    downsampled_lbc = lbc.contracted_point_cloud.voxel_down_sample(voxel_size = ct_args["centreline_down_sample"])
    downsampled_coords = np.asarray(downsampled_lbc.points) + local_shift

    print(f"downsampled the centreline cloud in {time() - s:.4f}s")
    s = time()

    # perform connected component analysis to get rid of erroneous data points that sometimes appear 
    cc0_downsampled_coords = return_largest_component(downsampled_coords, 
                                    eps = 0.5,
                                    min_component_size= 10)

    print(f"found largest component in {time() - s:.4f}s")
    s = time()

    mst = mist.GetMST(x=cc0_downsampled_coords[:, 0], 
                    y=cc0_downsampled_coords[:, 1], 
                    z=cc0_downsampled_coords[:, 2])

    degree, _, _, _, edge_index, branch_index = mst.get_stats(include_index=True, k_neighbours= 5)

    print(f"computed minimum spanning tree in {time() - s:.4f}s")
    s = time()
    fullG = nx.graph.Graph(edge_index.T.tolist())
    print(f"created graph instance of the centreline in  {time() - s:.4f}s")
    s = time()

    for idx in fullG.nodes():
        fullG.nodes[idx]['pos'] = cc0_downsampled_coords[idx].T
        fullG.nodes[idx]["branch"] = []

    for c, branch in enumerate(branch_index):
        for idx in branch:
            fullG.nodes[idx]["branch"].append(c)
    print(f"added node coordinates to graph in  {time() - s:.4f}s")
    s = time()

    # find the longest path on the tree: our centreline. 

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
    best_path = np.array(nx.shortest_path(fullG, leaf[i], leaf[j])).astype(int)

    return dict(sparse_cloud=cc0_downsampled_coords,
                mst_graph = fullG,
                centreline= best_path,
                skeleton_cloud_size=np.asarray(lbc.contracted_point_cloud.points).shape[0])