// -*- mode: c++; tab-width: 4; indent-tabs-mode: t; eval: (progn (c-set-style "stroustrup") (c-set-offset 'innamespace 0)); -*-
//
// bspliner.cpp
//
// Approximate a polyline by a B-spline
//

#include <Mathematics/BSplineCurveFit.h>

#include <cmath>
#include <iostream>
#include <vector>

std::vector<double> readPath(const char * fname) {
	std::vector<double> path;
	uint32_t size;
	FILE * fd = std::fopen(fname, "rb");
	if (!fd || fread(&size, sizeof(size), 1, fd) != 1) {
		std::cerr << "Error reading size from '" << fname << "'" << std::endl;
		exit(9);
	}
	path.resize(size);
	if (fread(path.data(), sizeof(double), size, fd) != size) {
		std::cerr << "Error reading data from '" << fname << "'" << std::endl;
		exit(9);
	}
	fclose(fd);
	return path;
}

void writePath(const char * fname, const double * data, uint32_t size) {
	FILE * fd = std::fopen(fname, "wb");
	fwrite(&size, sizeof(size), 1, fd);
	fwrite(data, sizeof(double), size, fd);
	fclose(fd);
}

int main(int argc, const char * argv[]) {
	if (argc != 4) {
		std::cerr << "Usage: bspliner <numControls> polyline.bin bspline.bin"
				  << std::endl;
		exit(9);
	}
	int numControls = std::strtol(argv[1], nullptr, 10);
	std::vector<double> samples = readPath(argv[2]);
	std::cerr << "Path has " << samples.size() / 3 << " vertices." << std::endl;
	gte::BSplineCurveFit<double> fit(3, samples.size() / 3, samples.data(), 3,
									 numControls);
	const double * p = fit.GetControlData();
	writePath(argv[3], p, 3 * numControls);
	return 0;
}
