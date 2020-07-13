/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#ifndef COLLISION_HPP
#define COLLISION_HPP

#include "simulation.hpp"
#include "cloth.hpp"
#include "constraint.hpp"
using torch::Tensor;

#ifndef FAST_MODE

void collision_response (Simulation &sim, std::vector<Mesh*> &meshes,
                         const std::vector<Constraint*> &cons,
                         const std::vector<Mesh*> &obs_meshes,bool verbose=false);
namespace CO {
struct Impact {
    enum Type {VF, EE} type;
    Tensor t;
    Node *nodes[4];
    Tensor w[4];
    Tensor n;

    std::vector<Node*>  imp_nodes;
    std::vector<Tensor> imp_Js;     // partial pdot /partial qdot
    std::vector<int>    mesh_num; // 0 cloth  >0 rigid number
    Impact () {}
    //Impact (Type type, const Node *n0, const Node *n1, const Node *n2,
    //        const Node *n3): type(type) {
    //    nodes[0] = (Node*)n0;
    //    nodes[1] = (Node*)n1;
    //    nodes[2] = (Node*)n2;
    //    nodes[3] = (Node*)n3;
    //}
};

struct ImpactZone {
    vector<Node*> nodes;
    vector<Impact> impacts;
    vector<double> w, n;
    vector<Tensor> xs;
    vector<Tensor> x0s;
    vector<int> mesh_num;
    vector<int> node_index;
    bool active;
    int nvar;
};
} //namespace CO

#else



void collision_response (Simulation &sim, std::vector<Mesh*> &meshes,
                         const std::vector<Constraint*> &cons,
                         const std::vector<Mesh*> &obs_meshes,bool verbose=false);
namespace CO {
struct Impact {
    enum Type {VF, EE} type;
    Tensor t;
    Node *nodes[4];
    Tensor w[4];
    Tensor n;

    double d_w[4];
    double d_n[3];

    std::vector<Node*>  imp_nodes;
    std::vector<Tensor> imp_Js;     // partial pdot /partial qdot
    std::vector<int>    mesh_num; // 0 cloth  >0 rigid number
    // d2t
    double  d_Js[4][3][6];

    Impact () {}
    //Impact (Type type, const Node *n0, const Node *n1, const Node *n2,
    //        const Node *n3): type(type) {
    //    nodes[0] = (Node*)n0;
    //    nodes[1] = (Node*)n1;
    //    nodes[2] = (Node*)n2;
    //    nodes[3] = (Node*)n3;
    //}
};

struct ImpactZone {
    vector<Node*> nodes;
    vector<Impact> impacts;
    vector<double> w, n;
    vector<Tensor> xs;
    vector<Tensor> x0s;
    vector<int> mesh_num;
    vector<int> node_index;
    bool active;
    int nvar;
};
} //namespace CO

//t2a
inline void t2a_get_subvec(double *ans, const double *a, int index){
  for (int i = 0; i < 3; i++) ans[i] = a[index+i];
}

inline void t2a_sub(double *ans, const double *a, const double *b){
  for (int i = 0; i < 3; i++) ans[i] = a[i]-b[i];
}

inline void t2a_add(double *ans, const double *a, const double *b){
  for (int i = 0; i < 3; i++) ans[i] = a[i]+b[i];
}


inline double t2a_dot(const double *a, const double *b){
  double ans = 0;
  for (int i = 0; i < 3; i++) ans += a[i]*b[i];
  return ans;
}

inline void t2a_matrix_mul_vec(double *ans, const double m[][3], const double *v){
  for (int i = 0; i < 3; i++) {
    ans[i] = 0;
    for (int j = 0; j < 3; ++j) ans[i] += m[i][j] * v[j];
  }
}

inline void t2a_vec_mul_matrix(double *ans, const double *v, const double m[][3]){
  for (int i = 0; i < 3; i++) {
    ans[i] = 0;
    for (int j = 0; j < 3; ++j) ans[i] += v[j] * m[j][i];
  }
}

inline void t2a_mul_scalar(double *ans, const double *a, double b){
  for (int i = 0; i < 3; i++) ans[i] = a[i]*b;
}

inline void t2a_set_subvec(double *ans, int index, const double *b){
  for (int i = 0; i < 3; i++) ans[i+index] = b[i];
}

inline void t2a_cross(double *w, const double *u, const double *v){
  w[0] = u[1]*v[2] - u[2]*v[1]; w[1] = u[2]*v[0] - u[0]*v[2]; w[2] = u[0]*v[1] - u[1]*v[0];
}
 
inline double t2a_stp(double *u, double *v, double *w) {
  double temp[3];
  t2a_cross(temp, v, w);
  return t2a_dot(u, temp);
}

inline double t2a_norm2(double *x) {return t2a_dot(x,x);}

inline double t2a_l2norm(double *x) {
  return std::sqrt(t2a_dot(x,x));
}


inline void t2a_normalize(double *x) {
  double l = t2a_l2norm(x);
  if ((l==0)) return;

  for (int i = 0; i < 3; i++) x[i] /= l;
}





#endif

#endif
