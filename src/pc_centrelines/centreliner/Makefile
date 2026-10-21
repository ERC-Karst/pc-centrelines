#
# Makefile for voxelize
#

CXXFLAGS += -O2 -g -Wall --std=c++20
CXXFLAGS += -Iclibs

all: voxelize bspliner

format:
	clang-format -i *.cpp
	black *.py

check:
	ruff *.py

voxelize: voxelize.cpp

bspliner: bspliner.cpp
