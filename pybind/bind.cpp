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
// std::vector<double> tovec(Vec3 a);
std::vector<double> tovec(Tensor a) {double *p = a.data<double>();return vector<double>(p,p+3);}

int msim (int argc, vector<string> argv);

namespace CO {
vector<Tensor> apply_inelastic_projection_forward(Tensor xold, Tensor ws, Tensor ns, ImpactZone *zone);
vector<Tensor> apply_inelastic_projection_backward(Tensor dldx_tn, Tensor ans_tn, Tensor q_tn, Tensor r_tn, Tensor lam_tn, Tensor sm1_tn, Tensor legals_tn, ImpactZone *zone);
}


vector<Tensor> solve_ixns_forward(Tensor xold, Tensor bs, Tensor ns, vector<Ixn> &ixns);
vector<Tensor> solve_ixns_backward(Tensor dldx_tn, Tensor ans_tn, Tensor q_tn, Tensor r_tn, Tensor lam_tn, Tensor sm1_tn, Tensor legals_tn, vector<Ixn> &ixns);

namespace SO {
vector<Tensor> solve_ixns_forward(Tensor xold, Tensor bs, Tensor ns, vector<Ixn> &ixns);
vector<Tensor> solve_ixns_backward(Tensor dldx_tn, Tensor ans_tn, Tensor q_tn, Tensor r_tn, Tensor lam_tn, Tensor sm1_tn, Tensor legals_tn, vector<Ixn> &ixns);
}

Tensor solve_cubic_forward(Tensor aa, Tensor bb, Tensor cc, Tensor dd);
vector<Tensor> solve_cubic_backward(Tensor dldz, Tensor ans, Tensor a, Tensor b, Tensor c, Tensor d);
Tensor taucs_linear_solve_forward (Tensor A, Tensor b, vector<pair<int, int> > indices);
vector<Tensor> taucs_linear_solve_backward (Tensor dldz, vector<pair<int, int> > indices, Tensor ans, Tensor A, Tensor b);

void init_physics (const string &json_file, string outprefix,
                   bool is_reloading);
void init_resume(const vector<string> &args);
void sim_step();
Simulation &get_sim();
void load_obj (Mesh &mesh, const string &filename);
void delete_mesh (Mesh &mesh);
void add_external_forces (const Cloth &cloth, const Tensor &gravity,
                          const Wind &wind, Tensor &fext,
                          Tensor &Jext);
void compute_ms_data(Mesh &mesh);

void test() {
	Tensor a = torch::ones({3}, TNOPT);
	cout << a << endl;
	double *d = a.cpu().data<double>();
	cout << d[0] << endl;
	vector<double> p(d,d+3);
	cout << p[0] << endl;
}

auto REF = py::return_value_policy::reference;
auto CPY = py::return_value_policy::copy;

PYBIND11_MAKE_OPAQUE(std::vector<Cloth>);
PYBIND11_MAKE_OPAQUE(std::vector<Obstacle>);
PYBIND11_MAKE_OPAQUE(std::vector<Cloth::Material*>);
PYBIND11_MAKE_OPAQUE(std::vector<Node*>);

PYBIND11_MODULE(arcsim, m){
	m.def("msim", &msim);
	m.def("test", &test);
	m.def("delete_mesh", &delete_mesh);
	m.def("add_external_forces",&add_external_forces);
	m.def("compute_ms_data",&compute_ms_data);
	py::class_<CO::ImpactZone>(m, "ImpactZone");
	m.def("apply_inelastic_projection_forward",&CO::apply_inelastic_projection_forward, CPY);
	m.def("apply_inelastic_projection_backward",&CO::apply_inelastic_projection_backward, CPY);
	py::class_<Ixn>(m, "Ixn");
	m.def("solve_ixns_forward",&solve_ixns_forward, CPY);
	m.def("solve_ixns_backward",&solve_ixns_backward, CPY);
	py::class_<SO::Ixn>(m, "SO_Ixn");
	m.def("SO_solve_ixns_forward",&SO::solve_ixns_forward, CPY);
	m.def("SO_solve_ixns_backward",&SO::solve_ixns_backward, CPY);
	m.def("solve_cubic_forward",&solve_cubic_forward);
	m.def("solve_cubic_backward",&solve_cubic_backward);
	m.def("taucs_linear_solve_forward",&taucs_linear_solve_forward);
	m.def("taucs_linear_solve_backward",&taucs_linear_solve_backward);

	m.def("init_physics",&init_physics);
	m.def("sim_step",&sim_step);
	m.def("load_obj",&load_obj);
	py::class_<Simulation>(m, "Simulation")
		.def_readwrite("cloths",&Simulation::cloths, REF)
		.def_readwrite("obstacles",&Simulation::obstacles, REF)
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
	py::bind_vector<std::vector<Obstacle> >(m, "VObstacle");
	py::class_<Cloth> cloth(m, "Cloth");
	cloth
		.def_readwrite("materials",&Cloth::materials, REF)
		.def_readwrite("mesh",&Cloth::mesh, REF)
		;

	py::class_<Obstacle> obstacle(m, "Obstacle");
	obstacle
		.def_readwrite("curr_state_mesh",&Obstacle::curr_state_mesh, REF)
		;

	py::bind_vector<std::vector<Cloth::Material*> >(m, "VMatP");
	py::class_<Cloth::Material>(cloth, "Material")
		.def_readwrite("densityori",&Cloth::Material::densityori, REF)
		.def_readwrite("stretchingori",&Cloth::Material::stretchingori, REF)
		.def_readwrite("stretching",&Cloth::Material::stretching, REF)
		.def_readwrite("bendingori",&Cloth::Material::bendingori, REF)
		;
	py::class_<Mesh>(m, "Mesh")
		.def(py::init<>())
		.def_readwrite("dummy_node",&Mesh::dummy_node, REF)
		.def_readwrite("nodes",&Mesh::nodes, REF)
		;
	py::bind_vector<std::vector<Node*> >(m, "VNodeP");
	py::class_<Node>(m, "Node")
		.def_readwrite("x",&Node::x, REF)
		.def_readwrite("v",&Node::v, REF)
		.def_readwrite("m",&Node::m, REF)
		;
	// py::class_<Vec3>(m, "Vec3")
	// 	.def("__getitem__", [](const Vec3 &s, size_t i) {
 //            return s[i];
 //        })
 //         .def("__setitem__", [](Vec3 &s, size_t i, double v) {
 //            s[i] = v;
 //        });
	m.def("tovec",&tovec);
	m.def("get_sim",&get_sim, REF);
}
