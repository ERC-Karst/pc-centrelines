# Center line computation

This code computes a center line for a cave conduit.

It needs a *closed* mesh of the conduit as input.

It also needs the entrance and exit points of the conduit.

## Structure

There are two C++ programs, which must be compiled first (just say
`make`).  The actual computation is then done by the Python script
`centerline.py`, which runs the C++ programs as needed.  See below for
how to run that.

### voxelize

The C++ program `voxelize` computes the intersection of a regular 3D
grid with the conduit.  In other words, it returns the set of all grid
points that lie inside the conduit.  It also returns the set of all
vertices of the mesh - those are later used for computing distances to
the wall.

Run it as
```
voxelize 800 conduit.ply voxels.vxl
```

The first argument is the grid size in millimeters.  The conduit mesh
can be in PLY or STL format.

The program writes the set of grid points and vertices to `voxels.vxl`
in some binary format that is easy to read from Python.


### bspliner

The C++ program `bspliner` fits a cubic B-pspline to a given polygonal
curve in 3D.

Run it as
```
bspliner <numControls> polyline.bin bspline.bin
```

The first argument is the desired number of control points of the
B-spline. It makes no sense to set this higher than half the number of
vertices of the polyline.

`polyline.bin` and `bspline.bin` are in a simple binary format that is
easy to read and write from Python.


## Running the center line computation

The actual center line computation is implemented in Python using
`networkx` for the graph algorithms.

You need a Python `venv` with the packages in `requirements.txt`.

Run it as
```
python3 centerline.py --conduit conduit.ply --output dir 
  --delta 800 --source x1,y1 --target x2,y2
```

The conduit can be in PLY or STL format.

All output files will be written to the directory `dir`, which is
created if it does not exist yet.

In principle, this should be a *closed* mesh, that is, before cutting
it open at the cave exits.  The reason is that the voxelizer uses the
mesh surface to determine which voxels lie inside the cave.  If you
run it with a mesh that has been cut open, it should still work, but it
will create more voxels, possibly many more, and therefore everything
will run more slowly.

`delta` is the grid size in millimeters used for the voxel grid.

`source` and `target` specify the coordinates of the endpoint of the
conduit.

Example:
```
python3 centerline.py --conduit ../pointclouds/VersChezLeBrandt_MainGallery_mesh_5cm_noshift.ply \
	--output test1 --delta 500 \
	--source 438.4,975.2,1147.2 \
	--target 568.8,820.8,1109.6
```

## How it works

