// -*- mode: c++; tab-width: 4; indent-tabs-mode: t; eval: (progn (c-set-style "stroustrup") (c-set-offset 'innamespace 0)); -*-
//
// voxelize.cpp
//
// Read a mesh of the conduit and rasterize it.
// Produces output in a simple binary format that contains
// 1. the set of grid points that lie in the interior of the conduit
// 2. all vertices of the conduit
//

#include <cmath>
#include <iostream>
#include <set>
#include <stl_reader.h>

#define TINYPLY_IMPLEMENTATION
#include <tinyply.h>

struct Point {
	float x, y, z;

	Point operator-(const Point & rhs) const {
		return Point{x - rhs.x, y - rhs.y, z - rhs.z};
	}

	bool operator==(const Point & rhs) const {
		return x == rhs.x && y == rhs.y && z == rhs.z;
	}

	float dot(const Point & rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }

	Point cross(const Point & rhs) const {
		return Point{
			y * rhs.z - z * rhs.y,
			z * rhs.x - x * rhs.z,
			x * rhs.y - y * rhs.x,
		};
	}

	bool operator<(const Point & rhs) const {
		if (x == rhs.x) {
			if (y == rhs.y)
				return z < rhs.z;
			else
				return y < rhs.y;
		} else
			return x < rhs.x;
	}
};

struct Stab {
	int x, y; // in mm, multiple of delta
	double z; // full-precision intersection with triangle
	bool up;
};

// 2D orientation: consider projection on xy-plane
// return 1 if r left of line from p to q
// 0 if on the line,
// -1 is right of the line
int orient(const Point & p, const Point & q, const Point & r) {
	Point a = q - p;
	Point b = r - p;
	double d = double(a.x) * double(b.y) - double(a.y) * double(b.x);
	return (d > 0) ? 1 : (d < 0) ? -1 : 0;
}

struct Triangle {
	std::array<Point, 3> p;

	void normalize() {
		if (orient(p[0], p[1], p[2]) < 0) std::swap(p[1], p[2]);
	}

	bool stabs(const Point & q) const {
		return orient(p[0], p[1], q) >= 0 && orient(p[1], p[2], q) >= 0
			   && orient(p[2], p[0], q) >= 0;
	}

	double interpolate(const Point & qq) const {
		Point p0 = p[0] - p[2];
		Point p1 = p[1] - p[2];
		Point q = qq - p[2];
		double det = p0.x * p1.y - p1.x * p0.y;
		// compute Barycentric coordinates
		double l1 = (p1.y * q.x - p1.x * q.y) / det;
		double l2 = (-p0.y * q.x + p0.x * q.y) / det;
		double l3 = 1 - l1 - l2;
		// Interpolate
		return l1 * p[0].z + l2 * p[1].z + l3 * p[2].z;
	}
};

class Voxelizer {
public:
	void readMeshSTL(const std::string & meshName);
	void readMeshPLY(const std::string & meshName);
	void addTriangle(const Triangle & t, bool up);
	void createVoxels();
	void writeVoxels(const char * voxelsName);

public:
	int32_t delta; // in mm
	std::vector<Stab> stabs;
	std::vector<int32_t> voxels;
	std::set<Point> vertices;
};

void Voxelizer::readMeshSTL(const std::string & meshName) {
	std::cerr << "Reading STL mesh" << std::endl;
	stl_reader::StlMesh<float, unsigned int> mesh(meshName.c_str());
	std::cerr << "Mesh has " << mesh.num_tris() << " triangles." << std::endl;

	for (size_t i = 0; i < mesh.num_tris(); ++i) {
		Triangle t;
		for (size_t j = 0; j < 3; ++j) {
			const float * c = mesh.tri_corner_coords(i, j);
			t.p[j] = Point{c[0], c[1], c[2]};
		}
		t.normalize();
		const float * n = mesh.tri_normal(i);
		bool up = n[2] > 0.0f; // normal points up
		addTriangle(t, up);
	}
}

void Voxelizer::readMeshPLY(const std::string & meshName) {
	std::cerr << "Reading PLY mesh" << std::endl;
	std::ifstream s(meshName, std::ios::binary);

	tinyply::PlyFile file;
	file.parse_header(s);

	for (const auto & c : file.get_comments()) std::cerr << "Comment: " << c << std::endl;
	for (const auto & c : file.get_info()) std::cerr << "Info: " << c << std::endl;

	std::shared_ptr<tinyply::PlyData> vertices, normals, faces;

	try {
		vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
	} catch (const std::exception & e) {
		std::cerr << "tinyply exception: " << e.what() << std::endl;
	}

	try {
		normals = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
	} catch (const std::exception & e) {
		std::cerr << "tinyply exception: " << e.what() << std::endl;
	}

	try {
		faces = file.request_properties_from_element("face", {"vertex_indices"});
	} catch (const std::exception & e) {
		std::cerr << "tinyply exception: " << e.what() << std::endl;
	}

	file.read(s);

	std::cout << "Read " << vertices->count << " total vertices " << std::endl;
	std::cout << "Read " << faces->count << " total faces (triangles) " << std::endl;

	if (vertices->t != tinyply::Type::FLOAT32 || normals->t != tinyply::Type::FLOAT32
		|| faces->t != tinyply::Type::INT32 || vertices->count != normals->count
		|| vertices->buffer.size_bytes() != vertices->count * 12
		|| normals->buffer.size_bytes() != vertices->count * 12
		|| faces->buffer.size_bytes() != faces->count * 12) {
		std::cerr << "Something unexpected happened:" << std::endl
				  << "Vertex type: " << int(vertices->t) << std::endl
				  << "Normals type: " << int(normals->t) << std::endl
				  << "Faces type: " << int(faces->t) << std::endl
				  << "Vertex buffer length: " << vertices->buffer.size_bytes()
				  << std::endl
				  << "Normals buffer length: " << normals->buffer.size_bytes()
				  << std::endl
				  << "Faces buffer length: " << faces->buffer.size_bytes() << std::endl;
		exit(9);
	}
	const Point * v = (Point *)vertices->buffer.get();
	const Point * n = (Point *)normals->buffer.get();
	const int32_t * triangles = (int32_t *)faces->buffer.get();

	size_t zeroTriangles = 0;
	size_t zeroEdges = 0;
	for (size_t i = 0; i < faces->count; ++i) {
		Triangle t;
		for (size_t j = 0; j < 3; ++j) { t.p[j] = v[triangles[3 * i + j]]; }
		if (t.p[0] == t.p[1] && t.p[1] == t.p[2]) {
			++zeroTriangles;
		} else if (t.p[0] == t.p[1] || t.p[1] == t.p[2] || t.p[0] == t.p[2]) {
			++zeroEdges;
		} else {
			t.normalize();
			// the PLY format contains vertex normals, not face normals,
			// so we need to compute the face normal and compare directions
			Point fn = (t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]);
			// use normal of t.p[0]
			Point vn = n[triangles[3 * i]];
			bool up = fn.dot(vn) > 0;
			addTriangle(t, up);
		}
	}
	std::cerr << "Ignored " << zeroTriangles << " singular triangles and " << zeroEdges
			  << " triangles with zero-length edge" << std::endl;
}

void Voxelizer::addTriangle(const Triangle & t, bool up) {
	vertices.insert(t.p[0]);
	vertices.insert(t.p[1]);
	vertices.insert(t.p[2]);
	const auto [minx, maxx] = std::minmax({t.p[0].x, t.p[1].x, t.p[2].x});
	const auto [miny, maxy] = std::minmax({t.p[0].y, t.p[1].y, t.p[2].y});
	int x0 = delta * int(std::floor(minx * 1000.0 / delta));
	int y0 = delta * int(std::floor(miny * 1000.0 / delta));
	for (int x = x0; x <= maxx * 1000.0f; x += delta) {
		for (int y = y0; y <= maxy * 1000.0f; y += delta) {
			Point q{x * 0.001f, y * 0.001f, 0.0f};
			if (t.stabs(q)) stabs.push_back({x, y, t.interpolate(q), up});
		}
	}
}

void Voxelizer::createVoxels() {
	std::cerr << "Sorting " << stabs.size() << " intersections" << std::endl;
	std::sort(stabs.begin(), stabs.end(), [](const Stab & a, const Stab & b) {
		if (a.x != b.x) return a.x < b.x;
		if (a.y != b.y) return a.y < b.y;
		return a.z < b.z;
	});
	std::cerr << "Creating voxels at " << delta << " mm interval" << std::endl;
	for (auto it = stabs.begin(); it != stabs.end();) {
		auto jt = it;
		while (jt != stabs.end() && jt->x == it->x && jt->y == it->y) ++jt;
		int z0 = delta * int(std::floor(it->z * 1000.0 / delta));
		bool inConduit = false;
		while (it < jt) {
			if (z0 < it->z * 1000.0f) {
				if (inConduit) {
					voxels.push_back(it->x);
					voxels.push_back(it->y);
					voxels.push_back(z0);
				}
				z0 += delta;
			} else {
				inConduit = !it++->up;
			}
		}
	}
	std::cerr << "Created " << voxels.size() / 3 << " voxels" << std::endl;
}

// all coordinates are written in millimeters
void Voxelizer::writeVoxels(const char * voxelsName) {
	std::cerr << "Writing '" << voxelsName << "': " << voxels.size() / 3 << " voxels and "
			  << vertices.size() << " vertices" << std::endl;
	FILE * fd = std::fopen(voxelsName, "wb");
	fwrite(&delta, sizeof(delta), 1, fd);
	int32_t size = voxels.size() / 3;
	fwrite(&size, sizeof(size), 1, fd);
	fwrite(voxels.data(), sizeof(int32_t), voxels.size(), fd);
	size = vertices.size();
	fwrite(&size, sizeof(size), 1, fd);
	std::vector<int32_t> vs;
	for (const Point & p : vertices) {
		vs.push_back(int32_t(std::round(1000.f * p.x)));
		vs.push_back(int32_t(std::round(1000.f * p.y)));
		vs.push_back(int32_t(std::round(1000.f * p.z)));
	}
	fwrite(vs.data(), sizeof(int32_t), vs.size(), fd);
	fclose(fd);
}

int main(int argc, const char ** argv) {
	if (argc != 4) {
		std::cerr << "Usage: voxelize size_in_mm mesh.{stl|ply} voxels.vxl" << std::endl;
		exit(9);
	}
	int delta = std::strtol(argv[1], nullptr, 10);
	std::string meshName = argv[2];
	const char * voxelsName = argv[3];

	Voxelizer v;
	v.delta = delta; // mm
	if (meshName.ends_with(".stl"))
		v.readMeshSTL(meshName);
	else
		v.readMeshPLY(meshName);
	v.createVoxels();
	v.writeVoxels(voxelsName);
	return 0;
}
