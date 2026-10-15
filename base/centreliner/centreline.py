#
# Compute a centerline for a conduit mesh
#

import struct
import sys
import time
import os
import subprocess
import argparse

import numpy as np
from numpy.linalg import norm
from scipy.spatial import KDTree
import networkx as nx
import ezdxf

import centreliner.spline as spline
import centreliner.simplify as simplify
from os.path import normpath


def runModule(module: str, args: list[str]) -> None:

    module_path = normpath(f"{module}.exe")

    if os.name == "nt":
        subprocess.run([f"{module_path}"] + args)
    else:
        subprocess.run([f"{module_path}"] + args)


# returns position in millimeters
def parsePosition(s: str) -> np.ndarray:
    f = s.split(",")
    if len(f) != 3:
        sys.stderr.write("Source and target must have 3 coordinates.\n")
        sys.exit(9)
    try:
        x = float(f[0])
        y = float(f[1])
        z = float(f[2])
    except Exception:
        sys.stderr.write("Coordinate cannot be parsed as a float.\n")
        sys.exit(9)
    return np.array([round(x * 1000), round(y * 1000), round(z * 1000)])


class Centreliner:

    def __init__(self, vxlName: str, delta: int):
        self.G = nx.Graph()
        with open(vxlName, "rb") as fd:
            data = fd.read()
            self.delta = struct.unpack("=i", data[:4])[0]
            assert self.delta == delta
            nVoxels = struct.unpack("=i", data[4:8])[0]
            indexVertices = 12 * nVoxels + 8
            for x, y, z in struct.iter_unpack("=iii", data[8:indexVertices]):
                self.G.add_node((x, y, z))
            nVertices = struct.unpack("=i", data[indexVertices : indexVertices + 4])[0]
            self.points = []
            for p in struct.iter_unpack("=iii", data[indexVertices + 4 :]):
                self.points.append(np.array(p))
            assert len(self.points) == nVertices
        self.tree = KDTree(self.points)
        sys.stderr.write(
            f"Read '{vxlName}': {len(self.points):,d} vertices and "
            + f"{len(self.G):,d} grid points,\n"
        )

    def addEdges(self):
        t1 = time.time()
        for n in self.G.nodes:
            x, y, z = n
            for dx in (0, self.delta):
                n1 = x + dx, y, z
                if n1 > n and n1 in self.G.nodes:
                    self.addEdge(n, n1)
            for dy in (-self.delta, 0, self.delta):
                n1 = x, y + dy, z
                if n1 > n and n1 in self.G.nodes:
                    self.addEdge(n, n1)
            for dz in (-self.delta, 0, self.delta):
                n1 = x, y, z + dz
                if n1 > n and n1 in self.G.nodes:
                    self.addEdge(n, n1)
        t2 = time.time()
        sys.stderr.write(f"Added edges in {t2 - t1:,.1f} sec.\n")

    def addEdge(self, pp, qq):
        p = np.array(pp)
        q = np.array(qq)
        x = 0.5 * (p + q)
        clearance = self.tree.query(x)[0]
        length = norm(p - q)
        self.G.add_edge(pp, qq, clearance=clearance, length=length)

    def findNearestNode(self, p: np.ndarray, verbose = False):
        nearest = None
        dist = 1e38
        for n in self.G.nodes:
            q = np.array(n)
            d = norm(p - q)
            if d < dist:
                dist = d
                nearest = n
        
        if verbose == True:
            print(f"distance: {dist:.1f} mm\n", "nearest: ", nearest)
        return nearest

    def computeCenterline(self, s, t) -> list[np.ndarray]:
        t0 = time.time()
        self.T = nx.maximum_spanning_tree(self.G, weight="clearance")
        t1 = time.time()
        sys.stderr.write(
            f"Computed maximum spanning tree for clearance in {t1 - t0:,.1f} sec.\n"
        )
        path = nx.shortest_path(self.T, source=s, target=t, weight="length")
        t2 = time.time()
        sys.stderr.write(
            f"Computed path in maximum spanning tree in {t2 - t1:,.1f} sec.\n"
        )
        return path
    
    def writeMSTEdges(self, fname: str):

        with open(fname, "w") as out:
            for p, q, _ in self.T.edges.data("length"):
                p = 0.001 * np.array(p)
                q = 0.001 * np.array(q)  # scale to meters

                out.write(f"{p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]}\n")
    

    # writes a point cloud: midpoint of each edge, together with clearance as scalar field
    def writeEdgesMidpoints(self, fname: str):
        with open(fname, "w") as out:
            for p, q, clearance in self.G.edges.data("clearance"):
                x = 0.0005 * (np.array(p) + np.array(q))  # scale to meters
                out.write(f"{x[0]} {x[1]} {x[2]} {clearance}\n")
    
    # writes a point cloud: midpoint of each edge, together with clearance as scalar field
    def writeEdges(self, fname: str):
        with open(fname, "w") as out:
            for p, q, clearance in self.G.edges.data("clearance"):
                p = 0.001 * np.array(p)
                q = 0.001 * np.array(q)  # scale to meters

                out.write(f"{p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]} {clearance}\n")


class Unzigzag:

    def __init__(self, path, nodes, delta: int):
        # for NN queries on the path
        self.tree = KDTree(path)
        self.delta = delta
        self.G = nx.Graph()
        for node in nodes:
            d = self.tree.query(np.array(node))[0]
            if d < 2 * delta:
                self.G.add_node(node)

    def addEdges(self):
        t1 = time.time()
        for n in self.G.nodes:
            x, y, z = n
            for dx in (0, self.delta):
                for dy in (-self.delta, 0, self.delta):
                    for dz in (-self.delta, 0, self.delta):
                        n1 = x + dx, y + dy, z + dz
                        if n1 > n and n1 in self.G.nodes:
                            self.addEdge(n, n1)
        t2 = time.time()
        print(f"Added edges in {t2 - t1:,.1f} sec.")

    def addEdge(self, pp, qq):
        p = np.array(pp)
        q = np.array(qq)
        length = norm(p - q)
        self.G.add_edge(pp, qq, length=length)

    def shortestPath(self, s, t) -> list[np.ndarray]:
        t0 = time.time()
        p = nx.shortest_path(
            self.G,
            source=s,
            target=t,
            weight="length",
        )
        t1 = time.time()
        print(f"Computed shortest path in {t1 - t0:,.1f} sec.")
        return p
    
    # writes a point cloud: midpoint of each edge, together with clearance as scalar field
    def writeEdgesMidpoints(self, fname: str):
        with open(fname, "w") as out:
            for p, q, clearance in self.G.edges.data("clearance"):
                x = 0.0005 * (np.array(p) + np.array(q))  # scale to meters
                out.write(f"{x[0]} {x[1]} {x[2]} {clearance}\n")
    
    # writes a set of edges
    def writeEdges(self, fname: str):
        with open(fname, "w") as out:
            for p, q, _ in self.G.edges.data("clearance"):
                p = 0.001 * np.array(p)
                q = 0.001 * np.array(q)  # scale to meters
                
                out.write(f"{p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]}\n")


def writePoints(fname: str, points: list[np.ndarray]):
    with open(fname, "w") as out:
        for p in points:
            out.write(f"{p[0]/1000.0} {p[1]/1000.0} {p[2]/1000.0} 255 0 0\n")


def writeDXF(
    fname: str,
    points: list[np.ndarray],
    segs: list[(np.ndarray, np.ndarray)],
    color,
) -> None:
    doc = ezdxf.new()
    doc.layers.add("points")
    doc.layers.add("segments")
    msp = doc.modelspace()
    for p in points:
        q = (0.001 * p[0], 0.001 * p[1], 0.001 * p[2])  # scale to meters
        e = msp.add_point(q, dxfattribs={"layer": "points"})
        e.dxf.color = ezdxf.colors.RED
    for p1, p2 in segs:
        q1 = (0.001 * p1[0], 0.001 * p1[1], 0.001 * p1[2])  # scale to meters
        q2 = (0.001 * p2[0], 0.001 * p2[1], 0.001 * p2[2])
        e = msp.add_polyline3d([q1, q2], dxfattribs={"layer": "segments"})
        e.dxf.color = color
    doc.saveas(fname)


def writeDXFPath(fname: str, path: list[np.ndarray], color=ezdxf.colors.GREEN) -> None:
    segs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    writeDXF(fname, path, segs, color)


def writeBinPath(fname: str, points: list[np.ndarray]):
    with open(fname, "wb") as out:
        out.write(struct.pack("=I", len(points) * 3))
        for q in points:
            out.write(struct.pack("ddd", q[0], q[1], q[2]))


def readBinPath(fname: str) -> list[np.ndarray]:
    with open(fname, "rb") as fd:
        data = fd.read()
        size = struct.unpack("=i", data[:4])[0]
        path = []
        for x, y, z in struct.iter_unpack("ddd", data[4:]):
            path.append(np.array([x, y, z]))
    if 3 * len(path) != size:
        raise RuntimeError(f"Failure reading '{fname}'")
    return path


def main():
    parser = argparse.ArgumentParser(
        prog="centerline.py",
        description="Compute a centerline for a meshed conduit.",
    )
    parser.add_argument(
        "--conduit",
        help="The meshed conduit, in STL or PLY format",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="directory for writing all output files",
        required=True,
    )

  
    parser.add_argument("--delta", help="voxel size in mm", type=int, required=True)
    parser.add_argument("--source", help="source position", type=str, required=True)
    parser.add_argument("--target", help="target position", type=str, required=True)
    parser.add_argument("--to-dxf", help="output dxf files", type=str, required=False, default = "False")

    args = parser.parse_args()

    delta = args.delta
    source = parsePosition(args.source)
    target = parsePosition(args.target)
    to_dxf = parsePosition(args.to_dxf)

    if to_dxf == "True":
        to_dxf = True
    else:
        to_dxf = False

    dir = args.output
    if dir.endswith("/"):
        dir = dir[:-1]

    if not os.path.exists(dir):
        sys.stderr.write(f"Creating directory '{dir}'\n")
        os.mkdir(dir)

    # run voxelize
    voxels = f"{dir}/voxels.vxl"
    runModule("voxelize", [str(delta), args.conduit, voxels])

    c = Centreliner(voxels, delta)
    c.addEdges()

    # to look at the grid and clearance
    # c.writeEdges(f"{dir}/edges.xyz")

    source1 = c.findNearestNode(source)
    target1 = c.findNearestNode(target)
    p = c.computeCenterline(source1, target1)
    if to_dxf:
        writeDXFPath(f"{dir}/rawpath.dxf", p)

    u = Unzigzag(p, c.G.nodes, c.delta)
    u.addEdges()
    p = u.shortestPath(source1, target1)

    # polygonal path
    p1 = [source] + [np.array(x) for x in p] + [target]
    if to_dxf:
        writeDXFPath(f"{dir}/centerline.dxf", p1)

    # simplified polygonal path
    p2 = simplify.douglasPeucker(delta, p1)
    if to_dxf:
        writeDXFPath(f"{dir}/centerline-simplified.dxf", p2)

    sys.stderr.write(f"Polyline has {len(p1)} vertices\n")
    sys.stderr.write(f"Simplified polyline has {len(p2)} vertices\n")

    # now fit cubic B-splines to p1 and p2
    writeBinPath(f"{dir}/path.bin", p1)
    writeBinPath(f"{dir}/path-simplified.bin", p2)

    runModule("bspliner", [str(len(p1) / 3), f"{dir}/path.bin", f"{dir}/bspline.bin"])
    runModule("bspliner", [str(len(p2) / 3), f"{dir}/path.bin", f"{dir}/bspline1.bin"])
    runModule(
        "bspliner",
        [
            str(len(p2) / 3),
            f"{dir}/path-simplified.bin",
            f"{dir}/bspline-simplified.bin",
        ],
    )

    cps = readBinPath(f"{dir}/bspline.bin")
    s1 = spline.bspline(cps)
    s2 = spline.subdivideBeziers(s1, 100)

    ps = [bezier[0] for bezier in s2]
    ps.append(s2[-1][-1])
    if to_dxf:
        writeDXFPath(f"{dir}/centerspline-fit.dxf", ps, ezdxf.colors.RED)

    # create B-spline directly from vertices of simplified path
    s3 = spline.bspline(p2)
    s4 = spline.subdivideBeziers(s3, 100)

    ps = [bezier[0] for bezier in s4]
    ps.append(s4[-1][-1])
    if to_dxf:
        writeDXFPath(f"{dir}/centerspline-direct.dxf", ps, ezdxf.colors.GREEN)

    cps = readBinPath(f"{dir}/bspline-simplified.bin")
    s5 = spline.bspline(cps)
    s6 = spline.subdivideBeziers(s5, 100)

    ps = [bezier[0] for bezier in s6]
    ps.append(s6[-1][-1])
    if to_dxf:
        writeDXFPath(f"{dir}/centerspline-fit-simplified.dxf", ps, ezdxf.colors.YELLOW)

    cps = readBinPath(f"{dir}/bspline1.bin")
    s7 = spline.bspline(cps)
    s8 = spline.subdivideBeziers(s7, 100)

    ps = [bezier[0] for bezier in s8]
    ps.append(s8[-1][-1])
    if to_dxf:
        writeDXFPath(f"{dir}/centerspline-fit1.dxf", ps, ezdxf.colors.WHITE)


if __name__ == "__main__":
    main()
