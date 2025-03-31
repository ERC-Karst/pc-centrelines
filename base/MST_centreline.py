import numpy as np
import sys
from os.path import normpath, exists, split, abspath
from subprocess import check_output

from stl import mesh
from base.utils import find_neighbours_with_knn, return_rings
from base.centreline import *
from centreline import simplify, spline

__all__ = ['return_MST_centrelines']

DELTA = 200


def return_MST_centrelines(mesh_path, voxset_path,
                          delta: float = DELTA) -> dict:
    
    
    conduit_name = split(mesh_path)[-1].strip(".stl")

    voxelize_path = abspath(normpath(f"./centerline/voxelize.exe"))
    mesh_path = abspath(normpath(mesh_path))
    voxset_path = abspath(normpath(voxset_path))

    # run the voxelizer module. 
    check_output(f"{voxelize_path} {delta} {mesh_path} {voxset_path}", shell=True)

    # instantiate a Centerliner class using the output voxel set and delta.
    print("delta: ", delta)
    c = Centerliner(voxset_path, delta)

    c.addEdges()

    # to look at the grid and clearance
    # c.writeEdges(f"{thedir}/edges.xyz")
    keep = True
    
    meshdata = mesh.Mesh.from_file(mesh_path)

    adj_tr_path = mesh_path.strip(".stl") + "_adjacent_triangles.txt"

    if exists(adj_tr_path) and keep:
        adjacent_triangles = np.loadtxt(adj_tr_path).astype(int)
    else:
        adjacent_triangles = find_neighbours_with_knn(meshdata)
        np.savetxt(adj_tr_path,adjacent_triangles, fmt="%.1d")

        
    rings = return_rings(meshdata, adjacent_triangles)

    print("inlet coordinates: ", rings["P1"]*1000)
    print("outlet coordinates: ", rings["P2"]*1000)

    source1 = c.findNearestNode(rings["P1"]*1000)
    target1 = c.findNearestNode(rings["P2"]*1000)

    p = c.computeCenterline(source1, target1)

    u = Unzigzag(p, c.G.nodes, c.delta)
    u.addEdges()
    p = u.shortestPath(source1, target1)

    p1 =[np.array(x) for x in p]

    if not np.allclose(rings["P1"]*1000 - p1[0], 0, atol=delta/2):
        p1.insert(0, np.round(rings["P1"]*1000, 1))
    if not np.allclose(rings["P2"]*1000 - p1[-1], 0, atol=delta/2):
        p1+=[np.round(rings["P2"]*1000, 1)]

    coords = np.array(p1)/1000

    centerlines = {}
    
    # save to file.
    np.savetxt(f"./tests/centrelines/MST_{conduit_name}_{delta}_raw_centreline.txt", coords, fmt= "%.3e")
    centerlines["raw"] = coords

    thedir = "./test1/"
    # polygonal path
    writeDXFPath(f"{thedir}/centerline.dxf", p1)
    print(p1)
    # simplified polygonal path
    p2 = simplify.douglasPeucker(DELTA, p1)
    writeDXFPath(f"{thedir}/centerline-simplified.dxf", p2)

    coords = np.array(p2)/1000
    # save to file.
    np.savetxt(f"./tests/centrelines/MST_{conduit_name}_{DELTA}_centerline-simplified.txt", coords, fmt= "%.3e")
    centerlines["simplified"] = coords


    sys.stderr.write(f"Polyline has {len(p1)} vertices\n")
    sys.stderr.write(f"Simplified polyline has {len(p2)} vertices\n")

    # now fit cubic B-splines to p1 and p2
    writeBinPath(f"{thedir}/path.bin", p1)
    writeBinPath(f"{thedir}/path-simplified.bin", p2)

    runModule("./centerline/bspliner", [str(len(p1) / 3), f"{thedir}/path.bin", f"{thedir}/bspline.bin"])
    runModule("./centerline/bspliner", [str(len(p2) / 3), f"{thedir}/path.bin", f"{thedir}/bspline1.bin"])
    runModule(
        "./centerline/bspliner",
        [
            str(len(p2) / 3),
            f"{thedir}/path-simplified.bin",
            f"{thedir}/bspline-simplified.bin",
        ],
    )

    cps = readBinPath(f"{thedir}/bspline.bin")
    s1 = spline.bspline(cps)
    s2 = spline.subdivideBeziers(s1, 100)

    ps = [bezier[0] for bezier in s2]
    ps.append(s2[-1][-1])
    writeDXFPath(f"{thedir}/centerspline-fit.dxf", ps, ezdxf.colors.RED)

    if len(p2) > 2:

        # create B-spline directly from vertices of simplified path
        s3 = spline.bspline(p2)
        s4 = spline.subdivideBeziers(s3, 100)

        ps = [bezier[0] for bezier in s4]
        ps.append(s4[-1][-1])
        writeDXFPath(f"{thedir}/centerspline-direct.dxf", ps, ezdxf.colors.GREEN)

        coords = np.array(ps)/1000
        # save to file.
        np.savetxt(f"./tests/centrelines/MST_{conduit_name}_{delta}_centerspline-direct.txt", coords, fmt= "%.3e")
        centerlines["centerspline_direct"] = coords


        cps = readBinPath(f"{thedir}/bspline-simplified.bin")
        s5 = spline.bspline(cps)
        s6 = spline.subdivideBeziers(s5, 100)

        ps = [bezier[0] for bezier in s6]
        ps.append(s6[-1][-1])
        writeDXFPath(f"{thedir}/centerspline-fit-simplified.dxf", ps, ezdxf.colors.YELLOW)

        cps = readBinPath(f"{thedir}/bspline1.bin")
        s7 = spline.bspline(cps)
        s8 = spline.subdivideBeziers(s7, 100)

        ps = [bezier[0] for bezier in s8]
        ps.append(s8[-1][-1])
        writeDXFPath(f"{thedir}/centerspline-fit1.dxf", ps, ezdxf.colors.WHITE)
    else:
        print("simplified polyline has fewer than 2 vertices. Impossible to simplify further.")
        s4 = spline.subdivideBeziers(s2, 100)

        ps = [bezier[0] for bezier in s4]
        ps.append(s4[-1][-1])
        writeDXFPath(f"{thedir}/centerspline-direct.dxf", ps, ezdxf.colors.GREEN)
        
        coords = np.array(ps)/1000
        # save to file.
        np.savetxt(f"./tests/centrelines/MST_{conduit_name}_{delta}_centerspline-direct.txt", coords, fmt= "%.3e")
        centerlines["centerspline_direct"] = coords

    return centerlines