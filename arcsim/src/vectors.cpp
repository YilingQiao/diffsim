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

#include "vectors.hpp"
using namespace std;
using torch::Tensor;

Tensor ptr2ten(int *a, int n) {
  vector<int> b;
  for (int i = 0; i < n; ++i)
    b.push_back(a[i]);
  return torch::tensor(b,torch::dtype(torch::kI32));
}
Tensor ptr2ten(double *a, int n) {
  vector<double> b;
  for (int i = 0; i < n; ++i)
    b.push_back(a[i]);
  return torch::tensor(b,TNOPT);
}
Tensor arr2ten(real_2d_array a) {
  int n = a.rows(), m = a.cols();
  vector<double> tmp;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      tmp.push_back(a[i][j]);
  Tensor ans = torch::tensor(tmp, TNOPT).reshape({n, m});
  return ans;
}
real_2d_array ten2arr(Tensor a) {
  int n = a.size(0), m = a.size(1);
  auto foo_a = a.accessor<double,2>();
  real_2d_array ans;
  ans.setlength(n, m);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      ans[i][j] = foo_a[i][j];
  return ans;
}
real_1d_array ten1arr(Tensor a) {
  int n = a.size(0);
  auto foo_a = a.accessor<double,1>();
  real_1d_array ans;
  ans.setlength(n);
  for (int i = 0; i < n; ++i)
    ans[i] = foo_a[i];
  return ans;
}

Tensor kronecker (const Tensor &t1, const Tensor &t2) {
    int t1_height = t1.size(0), t1_width = t1.size(1);
    int t2_height = t2.size(0), t2_width = t2.size(1);
    int out_height = t1_height * t2_height;
    int out_width = t1_width * t2_width;

    Tensor tiled_t2 = t2.repeat({t1_height, t1_width});
    Tensor expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat({1, t2_height, t2_width, 1})
          .reshape({out_height, out_width})
    );

    return expanded_t1 * tiled_t2;
}

Eig eigen_decomposition(const Tensor &A) {
  auto ans = torch::symeig(A,true);
  Eig eig;
  eig.Q = std::get<1>(ans);
  eig.l = std::get<0>(ans);
  if ((eig.l[0] > eig.l[1]).item<int>()) {
    eig.Q = eig.Q.flip({1});
    eig.l = eig.l.flip({0});
  }
	return eig;
}

SVD singular_value_decomposition (const Tensor &A) {
    SVD svd;
    auto ans = torch::svd(A, false);
    svd.U = std::get<0>(ans);
    svd.s = std::get<1>(ans);
    svd.Vt = std::get<2>(ans).t();
    return svd;
}
