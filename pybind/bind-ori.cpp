#include <string>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "collision.hpp"
#include "separate.hpp"
#include "separateobs.hpp"
#include "util.hpp"
#include "mesh.hpp"
#include "simulation.hpp"
#include <vector>
#include <pybind11/stl_bind.h>
#include "vectors.hpp"
using namespace std;
using namespace torch;

namespace py = pybind11;
std::vector<double> tovec(Vec3 a);
//std::vector<double> tovec(Tensor a) {double *p = a.data<double>();return vector<double>(p,p+3);}

void init_physics (const string &json_file, string outprefix,
                   bool is_reloading);
void init_resume(const vector<string> &args);
void sim_step();
Simulation &get_sim();
void load_obj (Mesh &mesh, const string &filename);
void delete_mesh (Mesh &mesh);
void compute_ms_data(Mesh &mesh);

auto REF = py::return_value_policy::reference;
auto CPY = py::return_value_policy::copy;

PYBIND11_MAKE_OPAQUE(std::vector<Cloth>);
PYBIND11_MAKE_OPAQUE(std::vector<Cloth::Material*>);
PYBIND11_MAKE_OPAQUE(std::vector<Node*>);

PYBIND11_MODULE(arcsim, m){
	m.def("delete_mesh", &delete_mesh);
	m.def("tovec", &tovec);

	m.def("init_physics",&init_physics);
	m.def("sim_step",&sim_step);
	m.def("load_obj",&load_obj);
	py::class_<Simulation>(m, "Simulation")
		.def_readwrite("cloths",&Simulation::cloths, REF)
		.def_readwrite("gravity",&Simulation::gravity, REF)
		.def_readwrite("wind",&Simulation::wind, REF)
		.def_readwrite("frame",&Simulation::frame, REF)
		.def_readwrite("time",&Simulation::time, REF)
		.def_readwrite("step",&Simulation::step, REF)
		;
	py::class_<Wind>(m, "Wind")
		.def_readwrite("density",&Wind::density, REF)
		.def_readwrite("velocity",&Wind::velocity, REF)
		.def_readwrite("drag",&Wind::drag, REF)
		;
	py::bind_vector<std::vector<Cloth> >(m, "VCloth");
	py::class_<Cloth> cloth(m, "Cloth");
	cloth
		.def_readwrite("materials",&Cloth::materials, REF)
		.def_readwrite("mesh",&Cloth::mesh, REF)
		;
	py::bind_vector<std::vector<Cloth::Material*> >(m, "VMatP");
	py::class_<Mesh>(m, "Mesh")
		.def(py::init<>())
		.def_readwrite("nodes",&Mesh::nodes, REF)
		;
	py::bind_vector<std::vector<Node*> >(m, "VNodeP");
	py::class_<Node>(m, "Node")
		.def_readwrite("x",&Node::x, REF)
		.def_readwrite("v",&Node::v, REF)
		.def_readwrite("m",&Node::m, REF)
		;
	py::class_<Vec3>(m, "Vec3")
		.def("__getitem__", [](const Vec3 &s, size_t i) {
            return s[i];
        })
         .def("__setitem__", [](Vec3 &s, size_t i, double v) {
            s[i] = v;
        });
	m.def("get_sim",&get_sim, REF);
}
