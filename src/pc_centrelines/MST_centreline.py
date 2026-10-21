#    Copyright (C) 2024-2026 by
#    Tanguy Racine <tanguy.racine@unine.ch>
#    All rights reserved.
#    MIT license.
#
"""
pc-centrelines MST centreline
=============

The voxel-based centrelining algorithm

"""

import numpy as np
import sys
from os.path import normpath, split, abspath
from subprocess import check_output

from pc_centrelines.centreliner.centreline import *
from pc_centrelines.centreliner.centreline import simplify, spline

__all__ = ['return_MST_centrelines']

DELTA = 100


def return_MST_centrelines(mesh_path,
                           voxset_path,
                           P1,
                           P2,
                           delta: float = DELTA,
                           verbose: bool = False,
                           to_dxf: bool = False,
                           demo: bool = False
                           ) -> dict:
    """A function encapsulating the voxel-based centrelining process from a mesh.
        It is a wrapper on the C++ voxeliser and b-spline executables.

    :param mesh_path: Path to a standard  .stl file containing a triangulated surface 
    :type mesh_path: str
    :param mesh_path: Path to the set of voxels representing the cave interior. 
    :type mesh_path: str
    :param P1: a numpy array containing the position of the source
    :type P1: np.ndarray
    :param P2: a numpy array containing the position of the target
    :type P2: np.ndarray
    :param verbose: whether to include print statements
    :type verbose: bool, optional
    :param to_dxf: whether to export the centrelines to dxf format
    :type to_dxf: bool, optional
    :param demo: whether to save intermediate centreline steps to text files, useful for demonstration figures.
    :type demo: bool, optional
    :return: A dictionary of the output centrelines
    :rtype: dict
    """

    # find out the conduit name and id
    conduit_name = split(mesh_path)[-1][:-4]
    id = split(split(mesh_path)[0])[-1].split("_")[0]
    thedir = f"../data//{id}_{conduit_name}//mst"

    # set up the path to voxelizer executable
    voxelize_path = abspath(
        normpath(f"../src/pc_centrelines/centreliner/voxelize.exe"))

    # set up the path of the mesh and voxsel set data
    mesh_path = abspath(normpath(mesh_path))
    voxset_path = abspath(normpath(voxset_path))

    # run the voxelizer module.
    check_output(
        f'"{voxelize_path}" {delta} "{mesh_path}" "{voxset_path}"', shell=True)

    # instantiate a Centerliner class using the output voxel set and delta.
    c = Centreliner(voxset_path, delta)

    c.addEdges()

    source1 = c.findNearestNode(P1*1000, verbose=verbose)
    target1 = c.findNearestNode(P2*1000, verbose=verbose)

    p = c.computeCenterline(source1, target1)

    if verbose == True:
        print("p, the centreline is of type: ",  type(p))
        print("delta, the voxel grid size is: ", type(c.delta))

    jagged_centreline = [np.array(x) for x in p]
    coords = np.array(jagged_centreline) / 1000
    # save to file.
    np.savetxt(
        f"{thedir}//{conduit_name}_{delta}_jagged_centreline.txt", coords, fmt="%.5e")

    # grow a buffer using 26-neighbour adjacency
    u = Unzigzag(p, c.G.nodes, c.delta)
    u.addEdges()

    if demo == True:
        # to look at the grid and clearance
        c.writeEdgesMidpoints(f"{thedir}//{conduit_name}_edges_midpoints.xyz")
        # to look at the grid and clearance
        c.writeEdges(f"{thedir}//{conduit_name}_edges.xyz")
        # for demonstration purposes, write the edges of the maximum spanning tree used.
        c.writeMSTEdges(f"{thedir}//{conduit_name}_MST_edges.xyz")
        # to look at the grid and clearance
        u.writeEdges(f"{thedir}//{conduit_name}_unzigzag_edges.xyz")

    # find the shortest path in the buffer zone
    p = u.shortestPath(source1, target1)
    p1 = [np.array(x) for x in p]

    if not np.allclose(P1*1000 - p1[0], 0, atol=delta/2):
        p1.insert(0, np.round(P1*1000, 1))
    if not np.allclose(P2*1000 - p1[-1], 0, atol=delta/2):
        p1 += [np.round(P2*1000, 1)]

    # these are now the coordinates of the raw centreline
    coords = np.array(p1)/1000

    centerlines = {}

    # save to file.
    np.savetxt(
        f"{thedir}//{conduit_name}_{delta}_raw_centreline.txt", coords, fmt="%.5e")
    centerlines["raw"] = coords

    # polygonal path
    if to_dxf:
        writeDXFPath(f"{thedir}/centerline.dxf", p1)
    # simplified polygonal path
    p2 = simplify.douglasPeucker(DELTA, p1)
    if to_dxf:
        writeDXFPath(f"{thedir}/centerline-simplified.dxf", p2)

    coords = np.array(p2)/1000
    # save to file.
    np.savetxt(
        f"{thedir}//{conduit_name}_{delta}_centerline-simplified.txt", coords, fmt="%.5e")
    centerlines["simplified"] = coords

    sys.stderr.write(f"Polyline has {len(p1)} vertices\n")
    sys.stderr.write(f"Simplified polyline has {len(p2)} vertices\n")

    # now fit cubic B-splines to p1 and p2
    writeBinPath(f"{thedir}/path.bin", p1)
    writeBinPath(f"{thedir}/path-simplified.bin", p2)

    runModule(
        "../src/pc_centrelines/centreliner/bspliner",
        [
            str(len(p1) / 3),
            f"{thedir}/path.bin",
            f"{thedir}/bspline.bin"
        ]
    )

    runModule(
        "../src/pc_centrelines/centreliner/bspliner",
        [
            str(len(p2) / 3),
            f"{thedir}/path.bin",
            f"{thedir}/bspline1.bin"
        ]
    )

    runModule(
        "../src/pc_centrelines/centreliner/bspliner",
        [
            str(len(p2) / 3),
            f"{thedir}/path-simplified.bin",
            f"{thedir}/bspline-simplified.bin",
        ]
    )

    # there is no need to simplify a line if it only has 3 control points.

    if len(p1) > 3:
        cps = readBinPath(f"{thedir}/bspline.bin")
        s1 = spline.bspline(cps)
        s2 = spline.subdivideBeziers(s1, 100)

        ps = [bezier[0] for bezier in s2]
        ps.append(s2[-1][-1])
        if to_dxf:
            writeDXFPath(f"{thedir}/centerspline-fit.dxf",
                         ps, ezdxf.colors.RED)

    if len(p2) > 2:

        # create B-spline directly from vertices of simplified path
        s3 = spline.bspline(p2)
        s4 = spline.subdivideBeziers(s3, 100)

        ps = [bezier[0] for bezier in s4]
        ps.append(s4[-1][-1])
        if to_dxf:
            writeDXFPath(f"{thedir}/centerspline-direct.dxf",
                         ps, ezdxf.colors.GREEN)

        coords = np.array(ps)/1000
        # save to file.
        np.savetxt(
            f"{thedir}/{conduit_name}_{delta}_centerspline-direct.txt", coords, fmt="%.6e")
        centerlines["centerspline_direct"] = coords

        cps = readBinPath(f"{thedir}/bspline-simplified.bin")
        s5 = spline.bspline(cps)
        s6 = spline.subdivideBeziers(s5, 100)

        ps = [bezier[0] for bezier in s6]
        ps.append(s6[-1][-1])
        if to_dxf:
            writeDXFPath(
                f"{thedir}/centerspline-fit-simplified.dxf", ps, ezdxf.colors.YELLOW)

        coords = np.array(ps)/1000
        # save to file.
        np.savetxt(
            f"{thedir}//{conduit_name}_{delta}_centerspline-fit-simplified.txt", coords, fmt="%.5e")
        centerlines["centerspline_fit-simplified"] = coords

        cps = readBinPath(f"{thedir}/bspline1.bin")
        s7 = spline.bspline(cps)
        s8 = spline.subdivideBeziers(s7, 100)

        ps = [bezier[0] for bezier in s8]
        ps.append(s8[-1][-1])
        if to_dxf:
            writeDXFPath(f"{thedir}/centerspline-fit1.dxf",
                         ps, ezdxf.colors.WHITE)

        coords = np.array(ps)/1000
        # save to file.
        np.savetxt(
            f"{thedir}//{conduit_name}_{delta}_centerspline-fit1.txt", coords, fmt="%.5e")
        centerlines["centerspline_fit1"] = coords
        centerlines["original_grid_points"] = len(c.G)

    else:
        if verbose is True:
            print(
                "simplified polyline has fewer than 2 vertices. Impossible to simplify further.")

        s4 = spline.subdivideBeziers(s2, 100)

        ps = [bezier[0] for bezier in s4]
        ps.append(s4[-1][-1])
        if to_dxf:
            writeDXFPath(f"{thedir}/centerspline-direct.dxf",
                         ps, ezdxf.colors.GREEN)

        coords = np.array(ps)/1000
        # save to file.
        np.savetxt(
            f"{thedir}//{conduit_name}_{delta}_centerspline-direct.txt", coords, fmt="%.5e")
        centerlines["centerspline_direct"] = coords

        centerlines["original_grid_points"] = len(c.G)

    return centerlines
