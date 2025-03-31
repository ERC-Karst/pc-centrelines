import numpy as np
import mistree_pp as mist
import networkx as nx

from os.path import exists
from stl import Mesh
from time import time
from base.utils import find_neighbours_with_knn, array_to_o3d, spatially_downsample, return_largest_component, return_rings
from pc_skeletor import LBC

__all__ = ['return_LBC_centreline']

CT_ARGS = dict(keep=True, keep_centreline=True, centreline_down_sample= 0.5)
LBC_ARGS = dict(init_attraction = 0.5, init_contraction = 0.5, down_sample = 0.5)

def return_LBC_centreline(mesh_filepath, 
                          adjacent_triangles_filepath, 
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

    meshdata = Mesh.from_file(mesh_filepath)



    keep = ct_args["keep"]

    if exists(adjacent_triangles_filepath) and keep:
        adjacent_triangles = np.loadtxt(adjacent_triangles_filepath).astype(int)
    else:
        adjacent_triangles = find_neighbours_with_knn(meshdata)
        np.savetxt(adjacent_triangles_filepath, adjacent_triangles, fmt="%.1d")

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

    coords = np.asarray(lbc.contracted_point_cloud.points) + local_shift
        
    downsampled_coords = spatially_downsample(coords, 
                                                min_distance=ct_args["centreline_down_sample"])

    print(f"downsampled the centreline cloud in {time() - s:.4f}s")
    s = time()

    # perform connected component analysis to get rid of erroneous data points that sometimes appear 
    cc0_downsampled_coords = return_largest_component(downsampled_coords, 
                                    octree_level = 10,
                                    min_component_size= 10)

    print(f"found largest component in {time() - s:.4f}s")
    s = time()

    mst = mist.GetMST(x=cc0_downsampled_coords[:, 0], 
                    y=cc0_downsampled_coords[:, 1], 
                    z=cc0_downsampled_coords[:, 2])

    _, _, _, _, edge_index, branch_index = mst.get_stats(include_index=True, k_neighbours= 5)

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

    # find the shortest path linking node extremities 
    # definition of extremities (from initial guesses and normals using the mesh. )
    rings = return_rings(meshdata, adjacent_triangles)
    
    # array view of coordinates and indices of the graph.
    indices = np.arange(0, fullG.number_of_nodes())
    coords = np.array([fullG.nodes[idx]["pos"] for idx in indices])


    start_node = indices[np.argmin(np.linalg.norm(rings["P1"] - coords, axis =1))]
    end_node = indices[np.argmin(np.linalg.norm(rings["P2"] - coords, axis =1))]
    # the shortest path linking our two extremities.


    centreline_node_path = nx.shortest_path(fullG, start_node, end_node)
    print(f"computed shortest path in  {time() - s:.4f}s")
    s = time()

    centreline = np.vstack([rings["P1"], coords[centreline_node_path], rings["P2"]])

    return dict(centreline=centreline, rings=rings)