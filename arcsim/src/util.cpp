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

#include "util.hpp"
#include "io.hpp"
#include "mesh.hpp"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
using namespace std;
using torch::Tensor;

inline string stringf (const string &format, ...) {
    char buf[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buf, 256, format.c_str(), args);
    va_end(args);
    return std::string(buf);
}

template <typename T> string name (const T *p) {
    stringstream ss;
    ss << setw(3) << setfill('0') << hex << ((size_t)p/sizeof(T))%0xfff;
    return ss.str();
}

ostream &operator<< (ostream &out, const Vert *vert) {
    out << "v:" << name(vert); return out;}

ostream &operator<< (ostream &out, const Node *node) {
    out << "n:" << name(node) << node->verts; return out;}

ostream &operator<< (ostream &out, const Edge *edge) {
    out << "e:" << name(edge) << "(" << edge->n[0] << "-" << edge->n[1] << ")"; return out;}

ostream &operator<< (ostream &out, const Face *face) {
    out << "f:" << name(face) << "(" << face->v[0] << "-" << face->v[1] << "-" << face->v[2] << ")"; return out;}
Tensor sgn(Tensor x){return (x>=0)*ONE-(x<0)*ONE;}
int solve_quadratic (Tensor a, Tensor b, Tensor c, Tensor x[2]) {
    // http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
    Tensor d = b*b - 4*a*c;
    if ((d < 0).item<int>()) {
        x[0] = -b/(2*a);
        return 0;
    }
    Tensor q = -(b + sgn(b).to(torch::kF64)*sqrt(d))/2;
    int i = 0;
    if ((abs(a) > 1e-12*abs(q)).item<int>())
        x[i++] = q/a;
    if ((abs(q) > 1e-12*abs(c)).item<int>())
        x[i++] = c/q;
    if (i==2 && (x[0] > x[1]).item<int>())
        swap(x[0], x[1]);
    return i;
}

int solve_quadratic (double a, double b, double c, double x[2]) {
    // http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
    double d = b*b - 4*a*c;
    if (d < 0) {
        x[0] = -b/(2*a);
        return 0;
    }
    double q = -(b + sqrt(d))/2;
    double q1 = -(b - sqrt(d))/2;
    int i = 0;
    if (abs(a) > 1e-12) {
        x[i++] = q/a;
        x[i++] = q1/a;
    } else {
        x[i++] = -c/b;
    }
    if (i==2 && x[0] > x[1])
        swap(x[0], x[1]);
    return i;
}

double newtons_method (double a, double b, double c, double d, double x0,
                       int init_dir) {
    if (init_dir != 0) {
        // quadratic approximation around x0, assuming y' = 0
        double y0 = d + x0*(c + x0*(b + x0*a)),
               ddy0 = 2*b + (x0+init_dir*1e-6)*(6*a);
        x0 += init_dir*sqrt(abs(2*y0/ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        double y = d + x0*(c + x0*(b + x0*a));
        double dy = c + x0*(2*b + x0*3*a);
        if (dy == 0)
            return x0;
        double x1 = x0 - y/dy;
        if (abs(x0 - x1) < 1e-6)
            return x0;
        x0 = x1;
    }
    return x0;
}

Tensor solve_cubic_forward(Tensor aa, Tensor bb, Tensor cc, Tensor dd){
    double a = aa.item<double>(), b = bb.item<double>();
    double c = cc.item<double>(), d = dd.item<double>();
    double xc[2], x[3];
    x[0]=x[1]=x[2]=-1;
    xc[0]=xc[1]=-1;
    int ncrit = solve_quadratic(3*a, 2*b, c, xc);
    if (ncrit == 0) {
        x[0] = newtons_method(a, b, c, d, xc[0], 0);
        return torch::tensor({x[0]},TNOPT);
    } else if (ncrit == 1) {// cubic is actually quadratic
        int nsol = solve_quadratic(b, c, d, x);
        return torch::tensor(vector<double>(x,x+nsol), TNOPT);
    } else {
        double yc[2] = {d + xc[0]*(c + xc[0]*(b + xc[0]*a)),
                        d + xc[1]*(c + xc[1]*(b + xc[1]*a))};
        int i = 0;
        if (yc[0]*a >= 0)
            x[i++] = newtons_method(a, b, c, d, xc[0], -1);
        if (yc[0]*yc[1] <= 0) {
            int closer = abs(yc[0])<abs(yc[1]) ? 0 : 1;
            x[i++] = newtons_method(a, b, c, d, xc[closer], closer==0?1:-1);
        }
        if (yc[1]*a <= 0)
            x[i++] = newtons_method(a, b, c, d, xc[1], 1);
        return torch::tensor(vector<double>(x,x+i), TNOPT);
    }
}

Tensor solve_cubic(Tensor a, Tensor b, Tensor c, Tensor d){
  py::object func = py::module::import("util_py").attr("solve_cubic");
  Tensor ans = func(a, b, c, d).cast<Tensor>();
  // Tensor ans = solve_cubic_forward(a, b, c, d);
  return ans;
}

vector<Tensor> solve_cubic_backward(Tensor dldz, Tensor ans, Tensor a, Tensor b, Tensor c, Tensor d) {
  Tensor dldd = dldz / (ans*(3*a*ans+2*b)+c);
  Tensor dldc = dldd * ans;
  Tensor dldb = dldc * ans;
  Tensor dlda = dldb * ans;
  return {dlda, dldb, dldc, dldd};
}

bool is_seam_or_boundary (const Vert *v) {
    return is_seam_or_boundary(v->node);
}

bool is_seam_or_boundary (const Node *n) {
    for (int e = 0; e < n->adje.size(); e++)
        if (is_seam_or_boundary(n->adje[e]))
            return true;
    return false;
}

bool is_seam_or_boundary (const Edge *e) {
    return !e->adjf[0] || !e->adjf[1] || edge_vert(e,0,0) != edge_vert(e,1,0);
}

bool is_seam_or_boundary (const Face *f) {
    return is_seam_or_boundary(f->adje[0])
        || is_seam_or_boundary(f->adje[1])
        || is_seam_or_boundary(f->adje[2]);
}

void debug_save_meshes (const vector<Mesh*> &meshvec, const string &name,
                        int n) {
    static map<string,int> savecount;
    if (n == -1)
        n = savecount[name];
    else
        savecount[name] = n;
    save_objs(meshvec, stringf("tmp/%s%04d", name.c_str(), n));
    savecount[name]++;
}

void debug_save_mesh (const Mesh &mesh, const string &name, int n) {
    static map<string,int> savecount;
    if (n == -1)
        n = savecount[name];
    else
        savecount[name] = n;
    save_obj(mesh, stringf("tmp/%s%04d.obj", name.c_str(), n));
    savecount[name]++;
}
