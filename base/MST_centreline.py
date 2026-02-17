import numpy as np
import sys
from os.path import normpath, exists, split, abspath
from subprocess import check_output

from stl import mesh
from base.utils import find_neighbours_with_knn, return_rings
from base.centreline import *
from base.centreline import simplify, spline

__all__ = ['return_MST_centrelines']

DELTA = 200


def return_MST_centrelines(mesh_path, voxset_path,P1, P2,
                          delta: float = DELTA, 
                          verbose = False,
                          to_dxf = False) -> dict:
    
    
    conduit_name = split(mesh_path)[-1][:-4]
    id = split(split(mesh_path)[0])[-1].split("_")[0]

    voxelize_path = abspath(normpath(f"./base/voxelize.exe"))
    mesh_path = abspath(normpath(mesh_path))
    voxset_path = abspath(normpath(voxset_path))

    # run the voxelizer module. 
    check_output(f'"{voxelize_path}" {delta} "{mesh_path}" "{voxset_path}"', shell=True)

    # instantiate a Centerliner class using the output voxel set and delta.
    c = Centerliner(voxset_path, delta)

    c.addEdges()

    # to look at the grid and clearance
    c.writeEdgesMidpoints(f"temp//{id}_{conduit_name}//mst//{conduit_name}_edges_midpoints.xyz")
    # to look at the grid and clearance
    c.writeEdges(f"temp//{id}_{conduit_name}//mst//{conduit_name}_edges.xyz")

    source1 = c.findNearestNode(P1*1000, verbose = verbose)
    target1 = c.findNearestNode(P2*1000, verbose = verbose)


    p = c.computeCenterline(source1, target1)

    # for demo purposes, write the edges of the maximum spanning tree used. 
    c.writeMSTEdges(f"temp//{id}_{conduit_name}//mst//{conduit_name}_MST_edges.xyz")

    jagged_centreline = [np.array(x) for x in p]
    coords = np.array(jagged_centreline) / 1000
    # save to file.
    np.savetxt(f"temp//{id}_{conduit_name}//mst//{conduit_name}_{delta}_jagged_centreline.txt", coords, fmt= "%.3e")

    u = Unzigzag(p, c.G.nodes, c.delta)
    u.addEdges()
    # to look at the grid and clearance
    u.writeEdges(f"temp//{id}_{conduit_name}//mst//{conduit_name}_unzigzag_edges.xyz")    

    p = u.shortestPath(source1, target1)

    p1 =[np.array(x) for x in p]

    if not np.allclose(P1*1000 - p1[0], 0, atol=delta/2):
        p1.insert(0, np.round(P1*1000, 1))
    if not np.allclose(P2*1000 - p1[-1], 0, atol=delta/2):
        p1+=[np.round(P2*1000, 1)]

    coords = np.array(p1)/1000

    centerlines = {}
    
    # save to file.
    np.savetxt(f"./temp//{id}_{conduit_name}//mst//{conduit_name}_{delta}_raw_centreline.txt", coords, fmt= "%.3e")
    centerlines["raw"] = coords

    thedir = f"./temp//{id}_{conduit_name}//mst"
    # polygonal path
    if to_dxf:
        writeDXFPath(f"{thedir}/centerline.dxf", p1)
    # simplified polygonal path
    p2 = simplify.douglasPeucker(DELTA, p1)
    if to_dxf:
         writeDXFPath(f"{thedir}/centerline-simplified.dxf", p2)

    coords = np.array(p2)/1000
    # save to file.
    np.savetxt(f"./temp//{id}_{conduit_name}//mst//{conduit_name}_{delta}_centerline-simplified.txt", coords, fmt= "%.3e")
    centerlines["simplified"] = coords


    sys.stderr.write(f"Polyline has {len(p1)} vertices\n")
    sys.stderr.write(f"Simplified polyline has {len(p2)} vertices\n")

    # now fit cubic B-splines to p1 and p2
    writeBinPath(f"{thedir}/path.bin", p1)
    writeBinPath(f"{thedir}/path-simplified.bin", p2)

    runModule("./base/bspliner", [str(len(p1) / 3), f"{thedir}/path.bin", f"{thedir}/bspline.bin"])
    runModule("./base/bspliner", [str(len(p2) / 3), f"{thedir}/path.bin", f"{thedir}/bspline1.bin"])
    runModule(
        "./base/bspliner",
        [
            str(len(p2) / 3),
            f"{thedir}/path-simplified.bin",
            f"{thedir}/bspline-simplified.bin",
        ],
    )

    # there is no need to simplify a line if it only has 3 control points. 

    if len(p1) > 3:
        cps = readBinPath(f"{thedir}/bspline.bin")
        s1 = spline.bspline(cps)
        s2 = spline.subdivideBeziers(s1, 100)

        ps = [bezier[0] for bezier in s2]
        ps.append(s2[-1][-1])
        if to_dxf:
            writeDXFPath(f"{thedir}/centerspline-fit.dxf", ps, ezdxf.colors.RED)

    if len(p2) > 2:

        # create B-spline directly from vertices of simplified path
        s3 = spline.bspline(p2)
        s4 = spline.subdivideBeziers(s3, 100)

        ps = [bezier[0] for bezier in s4]
        ps.append(s4[-1][-1])
        if to_dxf:
            writeDXFPath(f"{thedir}/centerspline-direct.dxf", ps, ezdxf.colors.GREEN)

        coords = np.array(ps)/1000
        # save to file.
        np.savetxt(f"./temp/centrelines/MST_{conduit_name}_{delta}_centerspline-direct.txt", coords, fmt= "%.3e")
        centerlines["centerspline_direct"] = coords


        cps = readBinPath(f"{thedir}/bspline-simplified.bin")
        s5 = spline.bspline(cps)
        s6 = spline.subdivideBeziers(s5, 100)

        ps = [bezier[0] for bezier in s6]
        ps.append(s6[-1][-1])
        if to_dxf:
            writeDXFPath(f"{thedir}/centerspline-fit-simplified.dxf", ps, ezdxf.colors.YELLOW)

        coords = np.array(ps)/1000
        # save to file.
        np.savetxt(f"temp//{id}_{conduit_name}//mst//{conduit_name}_{delta}_centerspline-fit-simplified.txt", coords, fmt= "%.3e")
        centerlines["centerspline_fit-simplified"] = coords


        cps = readBinPath(f"{thedir}/bspline1.bin")
        s7 = spline.bspline(cps)
        s8 = spline.subdivideBeziers(s7, 100)

        ps = [bezier[0] for bezier in s8]
        ps.append(s8[-1][-1])
        if to_dxf:
            writeDXFPath(f"{thedir}/centerspline-fit1.dxf", ps, ezdxf.colors.WHITE)

        
        coords = np.array(ps)/1000
        # save to file.
        np.savetxt(f"temp//{id}_{conduit_name}//mst//{conduit_name}_{delta}_centerspline-fit1.txt", coords, fmt= "%.3e")
        centerlines["centerspline_fit1"] = coords

    else:
        print("simplified polyline has fewer than 2 vertices. Impossible to simplify further.")
        s4 = spline.subdivideBeziers(s2, 100)

        ps = [bezier[0] for bezier in s4]
        ps.append(s4[-1][-1])
        if to_dxf:
            writeDXFPath(f"{thedir}/centerspline-direct.dxf", ps, ezdxf.colors.GREEN)
        
        coords = np.array(ps)/1000
        # save to file.
        np.savetxt(f"temp//{id}_{conduit_name}//mst//{conduit_name}_{delta}_centerspline-direct.txt", coords, fmt= "%.3e")
        centerlines["centerspline_direct"] = coords

    return centerlines