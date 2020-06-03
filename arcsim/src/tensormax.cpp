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

#include "tensormax.hpp"
#include "util.hpp"
using namespace std;
using torch::Tensor;

struct Disk {
    Tensor c;
    Tensor r;
    Disk (): c(torch::zeros({2},TNOPT)), r(ZERO) {}
    Disk (const Tensor &c, Tensor r): c(c), r(r) {}
};

// Welzl, Smallest enclosing disks..., 1991
Disk welzls_algorithm (const vector<Disk> &disks);

Tensor tensor_max (const vector<Tensor> &Ms) {
    int n = Ms.size();
    vector<Disk> disks;
    for (int i = 0; i < n; i++) {
        const Tensor &M = Ms[i];
        if ((trace(M) == 0).item<int>())
            continue;
        disks.push_back(Disk(torch::stack({(M[0][0]-M[1][1])/2, (M[0][1]+M[1][0])/2}),
                             (M[0][0]+M[1][1])/2));
    }
    Disk disk = welzls_algorithm(disks);
    return disk.c[0]*torch::tensor({1,0,0,-1},TNOPT).reshape({2,2})
         + disk.c[1]*torch::tensor({0,1,1,0},TNOPT).reshape({2,2})
         + disk.r*torch::tensor({1,0,0,1},TNOPT).reshape({2,2});
}

Disk minidisk (const vector<Disk> &P);
Disk b_minidisk (const vector<Disk> &P, const vector<Disk> &R);
Disk b_md (const vector<Disk> &R);

Disk welzls_algorithm (const vector<Disk> &disks) {
    return minidisk(disks);
}

bool enclosed (const Disk &disk0, const Disk &disk1);
template <typename T> T head (const vector<T> &v);
template <typename T> vector<T> tail (const vector<T> &v);
template <typename T> vector<T> cons (const T &x, const vector<T> &v);

Disk minidisk (const vector<Disk> &P) {
    if (P.empty())
        return Disk();
    Disk p = head(P);
    vector<Disk> P_ = tail(P);
    Disk D = minidisk(P_);
    if (enclosed(p, D))
        return D;
    else
        return b_minidisk(P_, vector<Disk>(1,p));
}

Disk b_minidisk (const vector<Disk> &P, const vector<Disk> &R) {
    if (P.empty() || R.size() == 3)
        return b_md(R);
    Disk p = head(P);
    vector<Disk> P_ = tail(P);
    Disk D = b_minidisk(P_, R);
    if (enclosed(p, D))
        return D;
    else
        return b_minidisk(P_, cons(p, R));
}

Disk apollonius (const Disk &disk1, const Disk &disk2, const Disk &disk3);

Disk b_md (const vector<Disk> &R) {
    if (R.empty())
        return Disk();
    else if (R.size() == 1)
        return head(R);
    else if (R.size() == 2) {
        Tensor d = norm(R[0].c - R[1].c);
        Tensor r = (R[0].r + d + R[1].r)/2;
        Tensor t = (r - R[0].r)/d;
        return Disk(R[0].c + t*(R[1].c - R[0].c), r);
    } else
        return apollonius(R[0], R[1], R[2]);
}

Disk apollonius (const Disk &disk1, const Disk &disk2, const Disk &disk3) {
    // nicked from http://rosettacode.org/mw/index.php?title=Problem_of_Apollonius&oldid=88212
#define DEFXYR(N) Tensor x##N = disk##N.c[0]; \
                  Tensor y##N = disk##N.c[1]; \
                  Tensor r##N = disk##N.r;
    DEFXYR(1); DEFXYR(2); DEFXYR(3);
#undef DEFXYR
    int s1 = 1, s2 = 1, s3 = 1;
    Tensor v11 = 2*x2 - 2*x1;
    Tensor v12 = 2*y2 - 2*y1;
    Tensor v13 = x1*x1 - x2*x2 + y1*y1 - y2*y2 - r1*r1 + r2*r2;
    Tensor v14 = 2*s2*r2 - 2*s1*r1;
    Tensor v21 = 2*x3 - 2*x2;
    Tensor v22 = 2*y3 - 2*y2;
    Tensor v23 = x2*x2 - x3*x3 + y2*y2 - y3*y3 - r2*r2 + r3*r3;
    Tensor v24 = 2*s3*r3 - 2*s2*r2;
    Tensor w12 = v12/v11;
    Tensor w13 = v13/v11;
    Tensor w14 = v14/v11;
    Tensor w22 = v22/v21-w12;
    Tensor w23 = v23/v21-w13;
    Tensor w24 = v24/v21-w14;
    Tensor P = -w23/w22;
    Tensor Q = w24/w22;
    Tensor M = -w12*P-w13;
    Tensor N = w14 - w12*Q;
    Tensor a = N*N + Q*Q - 1;
    Tensor b = 2*M*N - 2*N*x1 + 2*P*Q - 2*Q*y1 + 2*s1*r1;
    Tensor c = x1*x1 + M*M - 2*M*x1 + P*P + y1*y1 - 2*P*y1 - r1*r1;
    Tensor D = b*b-4*a*c;
    Tensor rs = (-b-sqrt(D))/(2*a);
    Tensor xs = M+N*rs;
    Tensor ys = P+Q*rs;
    return Disk(torch::stack({xs,ys}), rs);
}

bool enclosed (const Disk &disk0, const Disk &disk1) {
    return (norm(disk0.c-disk1.c) + disk0.r <= disk1.r + 1e-6).item<int>();
}

template <typename T> T head (const vector<T> &v) {
    return v.front();
}

template <typename T> vector<T> tail (const vector<T> &v) {
    vector<T> w(v.size()-1);
    for (int i = 0; i < w.size(); i++)
        w[i] = v[i+1];
    return w;
}

template <typename T> vector<T> cons (const T &x, const vector<T> &v) {
    vector<T> w(v.size()+1);
    w[0] = x;
    for (int i = 1; i < w.size(); i++)
        w[i] = v[i-1];
    return w;
}
