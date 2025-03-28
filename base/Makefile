#
# Makefile for voxelize
#

CXXFLAGS += -IGeometricTools/GTE
CXXFLAGS += -Istl_reader -Itinyply/source
CXXFLAGS += -O2 -g -Wall --std=c++20

all: voxelize bspliner

format:
	clang-format -i *.cpp
	black *.py

check:
	ruff *.py

voxelize: voxelize.cpp

bspliner: bspliner.cpp
