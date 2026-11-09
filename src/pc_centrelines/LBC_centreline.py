#    Copyright (C) 2024-2026 by
#    Tanguy Racine <tanguy.racine@unine.ch>
#    All rights reserved.
#    MIT license.
#
"""
pc-centrelines Laplacian based centreline
=============

The cloud contraction centrelining algorithm

"""

import numpy as np
import mistree as mist
import networkx as nx

from itertools import combinations
from time import time

from pc_centrelines.utils import array_to_o3d, return_largest_component
from pc_centrelines.laplacian import LBC

__all__ = ['return_LBC_centreline']


CT_ARGS = dict(keep_adjacent_triangles=True, centreline_down_sample=0.5)
LBC_ARGS = dict(init_attraction=0.5, init_contraction=0.5, down_sample=0.5)


def return_LBC_centreline(meshdata,
                          P1 : np.ndarray = None,
                          P2 : np.ndarray = None,
                          ct_args: dict = CT_ARGS,
                          lbc_args: dict = LBC_ARGS) -> dict:
    """A function encapsulating Laplacian-Based-contraction on a mesh

    :param mesh: An open3d mesh object
    :type mesh: _type_
    :param adjacent_triangles: a numpy array.
    :type adjacent_triangles: _type_
    :param ct_args: centreline downsampling and processing parameter, defaults to CT_ARGS
    :type ct_args: dict, optional
    :param lbc_args: Laplacian contraction dependent parameters, defaults to LBC_ARGS
    :type lbc_args: dictionary, optional
    :return: An array of 3d coordinates
    :rtype: np.ndarray
    """

    pcd = meshdata.centroids
    s = time()
    print(f"read file in {time() - s:.4f}s")

    s = time()

    # local shift
    local_shift = np.mean(pcd, axis=0)

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

    downsampled_lbc = lbc.contracted_point_cloud.voxel_down_sample(
        voxel_size=ct_args["centreline_down_sample"])
    downsampled_coords = np.asarray(downsampled_lbc.points) + local_shift

    print(f"downsampled the centreline cloud in {time() - s:.4f}s")
    s = time()

    # perform connected component analysis to get rid of erroneous data points that sometimes appear
    cc0_downsampled_coords = return_largest_component(downsampled_coords,
                                                      eps=0.5,
                                                      min_component_size=10)

    print(f"found largest component in {time() - s:.4f}s")
    s = time()

    mst = mist.GetMST(x=cc0_downsampled_coords[:, 0],
                      y=cc0_downsampled_coords[:, 1],
                      z=cc0_downsampled_coords[:, 2])

    degree, _, _, _, edge_index, branch_index = mst.get_stats(
        include_index=True, k_neighbours=5)

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
    leaves = np.argwhere(degree == 1).flatten().astype(int)

      # shift start and end points
    if P1 is not None and P2 is not None:
   
        print(f"this minimum spanning tree has {len(leaves)} leaves. P1 and P2 were defined.")

        # find the nearest index to P1.
        d1 = 1e10
        d2 = 1e10 
        idxP1 = None
        idxP2 = None
        # cycle over the graph nodes and track the indices minimising distance to P1 and P2.

        for leaf in leaves:
            print(f"checking leaf {leaf}")
            pos = fullG.nodes[leaf]["pos"]
            # euclidean distances to current node.
            dcur1 = np.minimum(d1, np.linalg.norm(pos-P1))
            dcur2 = np.minimum(d1, np.linalg.norm(pos-P2))

            if dcur1 < d1:
                d1 = dcur1
                idxP1 = leaf

            if dcur2 < d2:
                d2 = dcur2
                idxP2 = leaf
                
        best_path = np.array(nx.shortest_path(fullG, idxP1, idxP2)).astype(int)

    else:

        print(
            f"this minimum spanning tree has {len(leaves)} leaves, no P1 or P2 defined.\n Need to identify the best path.")

        # the shortest path linking our two extremities.
        best_shortest_path = 0
        best_comb = (0, 1)

        # iterate over combination of leaf indices to find the longest path (in terms of nodes) in the tree.
        for comb in combinations(list(range(len(leaves))), 2):
            i, j = comb
            centreline_path = nx.shortest_path_length(fullG, leaves[i], leaves[j])
            if centreline_path >= best_shortest_path:
                best_shortest_path = centreline_path
                best_comb = comb

        i, j = best_comb
        best_path = np.array(nx.shortest_path(fullG, leaves[i], leaves[j])).astype(int)
        
        P1 = cc0_downsampled_coords[best_path[0]]
        P2 = cc0_downsampled_coords[best_path[-1]]

    return dict(sparse_cloud=cc0_downsampled_coords,
                mst_graph=fullG,
                centreline=best_path,
                skeleton_cloud_size=np.asarray(lbc.contracted_point_cloud.points).shape[0],
                P1 = P1,  P2 = P2)
