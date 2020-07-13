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

#ifndef VECTORS_HPP
#define VECTORS_HPP

#include <torch/extension.h>
// #include <png.h>
#include "alglib/linalg.h"
#include "alglib/solvers.h"
#include <vector>
using namespace alglib;
using std::vector;
using torch::Tensor;

#include <cmath>
#include <iostream>

#define  EYE3  (torch::eye(3,TNOPT))
#define  TRU   (torch::ones({}, torch::dtype(torch::kU8)))
#define  ONE   (torch::ones({}, TNOPT))
#define  ZERO  (torch::zeros({}, TNOPT))
#define  DT_PI (torch::ones({}, TNOPT)*3.141592653589793)
#define  ZERO3 (torch::zeros({3}, TNOPT))
#define  ZERO6 (torch::zeros({6}, TNOPT))
#define  ZERO33 (torch::zeros({3,3}, TNOPT))
//#define  infinity (std::numeric_limits<double>::infinity()*ONE)

const torch::TensorOptions TNOPT = torch::dtype(torch::kF64);
// const Tensor EYE3=torch::eye(3,TNOPT);
// const Tensor TRU=torch::ones({}, torch::dtype(torch::kU8));
// const Tensor ONE=torch::ones({}, TNOPT);
// const Tensor ZERO=(torch::zeros({}, TNOPT));
// const Tensor DT_PI  = torch::ones({}, TNOPT)*3.141592653589793;
// const Tensor ZERO3=(torch::zeros({3}, TNOPT));
// const Tensor ZERO6=(torch::zeros({6}, TNOPT));
// const Tensor ZERO33=(torch::zeros({3,3}, TNOPT));
const Tensor infinity = std::numeric_limits<double>::infinity()*ONE;

inline Tensor perp (const Tensor &u) {return torch::stack({-u[1],u[0]});}

inline double clamp(double a,double b,double c){if (a<b)a=b;if (a>c)a=c;return a;}

Tensor arr2ten(real_2d_array a);
Tensor ptr2ten(int *a, int n);
Tensor ptr2ten(double *a, int n);
real_2d_array ten2arr(Tensor a);
real_1d_array ten1arr(Tensor a);
template <class T>
vector<T> ten2vec(Tensor a) {
  vector<T> ans;
  T *x = a.data<T>();
  int n = a.size(0);
  for (int i = 0; i < n; ++i)
    ans.push_back(x[i]);
  return ans;
}

Tensor kronecker (const Tensor &t1, const Tensor &t2);

inline Tensor rowmat (const Tensor &v) {return v.unsqueeze(0);}

inline Tensor get_subvec (const double *x, bool is_cloth, int start_dim) {
    return (is_cloth ? torch::tensor({x[start_dim+0], x[start_dim+1], x[start_dim+2]},TNOPT)
                    : torch::tensor({x[start_dim+0], x[start_dim+1], x[start_dim+2],
                            x[start_dim+3], x[start_dim+4], x[start_dim+5]},TNOPT));
  }

inline Tensor get_subvec (const double *x, int i) {
    return torch::tensor({x[i*3+0], x[i*3+1], x[i*3+2]},TNOPT);}
inline void set_subvec (double *x, int i, const Tensor &xi) {
    for (int j = 0; j < xi.sizes()[0]; j++) x[i+j] = xi[j].item<double>();}
inline void add_subvec (double *x, int i, const Tensor &xi) {
    for (int j = 0; j < 3; j++) x[i*3+j] += xi[j].item<double>();}

inline Tensor get_subvec (const Tensor x, int i) {
    return x.slice(0,i*3,i*3+3);}
inline void set_subvec (Tensor &x, int i, const Tensor &xi) {
    x.slice(0,i*3,i*3+3) = xi;}
inline void add_subvec (Tensor &x, int i, const Tensor &xi) {
    x.slice(0,i*3,i*3+3) +=xi;}//= x.slice(0,i*3,i*3+3) + xi;}

inline std::istream &operator>>(std::istream &in, Tensor &x) {
  double x0;
  in >> x0;
  x = x0*ONE;
  return in;
}

inline double sq (double x) {return x*x;}
inline Tensor sq (Tensor x) {return x*x;}

inline Tensor norm2(Tensor x) {return dot(x,x);}
inline Tensor wedge(Tensor u, Tensor v) {return u[0]*v[1]-u[1]*v[0];}

inline Tensor normalize(Tensor x) {
  Tensor l = torch::norm(x);
  if ((l==0).item<int>()) return x;
  return x / l;
}

inline Tensor stp(Tensor u, Tensor v, Tensor w) {
  return torch::dot(u, torch::cross(v, w));
}

struct Eig {
    Tensor Q;
    Tensor l;
};

Eig eigen_decomposition(const Tensor &A);

struct SVD {
    Tensor U;
    Tensor s;
    Tensor Vt;
};

SVD singular_value_decomposition (const Tensor &A);


#undef static_assert

#endif
