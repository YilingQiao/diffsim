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

#include "separate.hpp"

#include "collisionutil.hpp"
#include "geometry.hpp"
#include "io.hpp"
#include "magic.hpp"
#include "optimization.hpp"
#include "simulation.hpp"
#include "util.hpp"
#include <omp.h>

#include <vector>
#include "alglib/linalg.h"
#include "alglib/solvers.h"
using namespace std;
using namespace alglib;
#include <torch/torch.h>
using torch::Tensor;

static const int max_iter = 100;
static const Tensor &thickness = ::magic.projection_thickness;

static const vector<Mesh*> *old_meshes;
static vector<Tensor> xold;


// ostream &operator<< (ostream &out, const Ixn &ixn) {out << ixn.f0 << "@" << ixn.b0 << " " << ixn.f1 << "@" << ixn.b1 << " " << ixn.n; return out;}

vector<Ixn> find_intersections (const vector<AccelStruct*> &accs,
                                const vector<AccelStruct*> &obs_accs);

void solve_ixns (vector<Ixn> &ixns);

void separate (vector<Mesh*> &meshes, const vector<Mesh*> &old_meshes,
               const vector<Mesh*> &obs_meshes) {
    ::meshes = &meshes;
    ::old_meshes = &old_meshes;
    ::obs_meshes = &obs_meshes;
    ::xold = node_positions(meshes);
    vector<AccelStruct*> accs = create_accel_structs(meshes, false),
                         obs_accs = create_accel_structs(obs_meshes, false);
    vector<Ixn> ixns;
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        vector<Ixn> new_ixns = find_intersections(accs, obs_accs);
        if (new_ixns.empty())
            break;
        append(ixns, new_ixns);
        solve_ixns(ixns);
        for (int m = 0; m < meshes.size(); m++) {
            compute_ws_data(*meshes[m]);
            update_accel_struct(*accs[m]);
        }
    }
    if (iter == max_iter) {
        cerr << "Post-remeshing separation failed to converge!" << endl;
        debug_save_meshes(meshes, "meshes");
        debug_save_meshes(old_meshes, "oldmeshes");
        debug_save_meshes(obs_meshes, "obsmeshes");
        exit(1);
    }
    for (int m = 0; m < meshes.size(); m++) {
        compute_ws_data(*meshes[m]);
        update_x0(*meshes[m]);
    }
    destroy_accel_structs(accs);
    destroy_accel_structs(obs_accs);
}

static int nthreads = 0;
static vector<Ixn> *ixns = NULL;

void find_face_intersection (const Face *face0, const Face *face1);

vector<Ixn> find_intersections (const vector<AccelStruct*> &accs,
                                const vector<AccelStruct*> &obs_accs) {
    if (!::ixns) {
        ::nthreads = omp_get_max_threads();
        ::ixns = new vector<Ixn>[::nthreads];
    }
    for (int t = 0; t < ::nthreads; t++)
        ::ixns[t].clear();
    for_overlapping_faces(accs, obs_accs, ::thickness, find_face_intersection);
    vector<Ixn> ixns;
    for (int t = 0; t < ::nthreads; t++)
        append(ixns, ::ixns[t]);
    return ixns;
}

bool adjacent (const Face *face0, const Face *face1) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (face0->v[i]->node == face1->v[j]->node)
                return true;
    return false;
}

int major_axis (const Tensor &v) {
    return ((abs(v[0]) > abs(v[1])).item<double>() && (abs(v[0]) > abs(v[2])).item<double>()) ? 0
         : ((abs(v[1]) > abs(v[2])).item<double>()) ? 1 : 2;
}

bool face_plane_intersection (const Face *face, const Face *plane,
                              Tensor &b0, Tensor &b1) {
    // plane->n_tn = norm(plane);//
    Tensor x0 = plane->v[0]->node->x, n = plane->n;//normalize(cross(plane->v[1]->node->x_tn-x0,plane->v[2]->node->x_tn-x0));
    // cout << n << endl << endl;
    Tensor h[3];
    Tensor sign_sum = ZERO;
    for (int v = 0; v < 3; v++) {
        h[v] = dot(face->v[v]->node->x - x0, n);
        sign_sum = sign_sum + sign(h[v]);
    }
    if ((sign_sum == -3).item<int>() || (sign_sum == 3).item<int>())
        return false;
    int v0 = -1;
    for (int v = 0; v < 3; v++)
        if ((sign(h[v]) == -sign_sum).item<int>())
            v0 = v;
    Tensor t0 = h[v0]/(h[v0] - h[NEXT(v0)]), t1 = h[v0]/(h[v0] - h[PREV(v0)]);
    b0[v0] = 1 - t0;
    b0[NEXT(v0)] = t0;
    b0[PREV(v0)] = 0;
    b1[v0] = 1 - t1;
    b1[PREV(v0)] = t1;
    b1[NEXT(v0)] = 0;
    return true;
}

Tensor vf_clear_distance (const Face *face0, const Face *face1, const Tensor &d,
                          Tensor last_dist, Tensor &b0, Tensor &b1) {
    for (int v = 0; v < 3; v++) {
        const Tensor &xv = face0->v[v]->node->x, &x0 = face1->v[0]->node->x,
                   &x1 = face1->v[1]->node->x, &x2 = face1->v[2]->node->x;
        const Tensor &n = face1->n;
        Tensor h = dot(xv-x0, n), dh = dot(d, n);
        if ((h*dh >= 0).item<int>())
            continue;
        Tensor a0 = stp(x2-x1, xv-x1, d),
               a1 = stp(x0-x2, xv-x2, d),
               a2 = stp(x1-x0, xv-x0, d);
        if ((a0 <= 0).item<int>() || (a1 <= 0).item<int>() || (a2 <= 0).item<int>())
            continue;
        Tensor dist = -h/dh;
        if ((dist > last_dist).item<int>()) {
            last_dist = dist;
            b0 = ZERO3;
            b0[v] = 1;
            b1[0] = a0/(a0+a1+a2);
            b1[1] = a1/(a0+a1+a2);
            b1[2] = a2/(a0+a1+a2);
        }
    }
    return last_dist;
}

Tensor ee_clear_distance (const Face *face0, const Face *face1, const Tensor &d,
                          Tensor last_dist, Tensor &b0, Tensor &b1) {
    for (int e0 = 0; e0 < 3; e0++) {
        for (int e1 = 0; e1 < 3; e1++) {
            const Tensor &x00 = face0->v[e0]->node->x,
                       &x01 = face0->v[NEXT(e0)]->node->x,
                       &x10 = face1->v[e1]->node->x,
                       &x11 = face1->v[NEXT(e1)]->node->x;
            Tensor n = cross(normalize(x01-x00), normalize(x11-x10));
            Tensor h = dot(x00-x10, n), dh = dot(d, n);
            if ((h*dh >= 0).item<int>())
                continue;
            Tensor a00 = stp(x01-x10, x11-x10, d),
                   a01 = stp(x11-x10, x00-x10, d),
                   a10 = stp(x01-x00, x11-x00, d),
                   a11 = stp(x10-x00, x01-x00, d);
            if ((a00*a01 <= 0).item<int>() || (a10*a11 <= 0).item<int>())
                continue;
            Tensor dist = -h/dh;
            if ((dist > last_dist).item<int>()) {
                last_dist = dist;
                b0 = ZERO3;
                b0[e0] = a00/(a00+a01);
                b0[NEXT(e0)] = a01/(a00+a01);
                b1 = ZERO3;
                b1[e1] = a10/(a10+a11);
                b1[NEXT(e1)] = a11/(a10+a11);
            }
        }
    }
    return last_dist;
}

bool farthest_points (const Face *face0, const Face *face1, const Tensor &d,
                      Tensor &b0, Tensor &b1) {
    Tensor last_dist = ZERO;
    last_dist = vf_clear_distance(face0, face1, d, last_dist, b0, b1);
    last_dist = vf_clear_distance(face1, face0, -d, last_dist, b1, b0);
    last_dist = ee_clear_distance(face0, face1, d, last_dist, b0, b1);
    return (last_dist > 0).item<int>();
}

Tensor pos (const Face *face, const Tensor &b) {
    return b[0]*face->v[0]->node->x
         + b[1]*face->v[1]->node->x
         + b[2]*face->v[2]->node->x;
}

Tensor old_pos (const Face *face, const Tensor &b) {
    if (!is_free(face))
        return pos(face, b);
    Tensor u = b[0]*face->v[0]->u + b[1]*face->v[1]->u + b[2]*face->v[2]->u;
    int m;
    for (m = 0; m < ::meshes->size(); m++)
        if ((*::meshes)[m]->faces[face->index] == face)
            break;
    Face *old_face = get_enclosing_face(*(*::old_meshes)[m], u);
    Tensor old_b = get_barycentric_coords(u, old_face);
    return pos(old_face, old_b); //TODO: old face tn????
}

bool intersection_midpoint (const Face *face0, const Face *face1,
                            Tensor &b0, Tensor &b1) {
    if ((norm2(cross(face0->n, face1->n)) < 1e-12).item<int>())
        return false;
    Tensor b00, b01, b10, b11;
    b00 = ZERO3;
    b01 = ZERO3;
    b10 = ZERO3;
    b11 = ZERO3;
    bool ix0 = face_plane_intersection(face0, face1, b00, b01),
         ix1 = face_plane_intersection(face1, face0, b10, b11);
    if (!ix0 || !ix1)
        return false;
    int axis = major_axis(cross(face0->n, face1->n));
    Tensor a00 = pos(face0, b00)[axis], a01 = pos(face0, b01)[axis],
           a10 = pos(face1, b10)[axis], a11 = pos(face1, b11)[axis];
    Tensor amin = max(min(a00, a01), min(a10, a11)),
           amax = min(max(a00, a01), max(a10, a11)),
           amid = (amin + amax)/2;
    if ((amin > amax).item<int>())
        return false;
    b0 = (a01==a00).item<int>() ? b00 : b00 + (amid-a00)/(a01-a00)*(b01-b00);
    b1 = (a11==a10).item<int>() ? b10 : b10 + (amid-a10)/(a11-a10)*(b11-b10);
    // cout << b0 << endl;
    return true;
}

void find_face_intersection (const Face *face0, const Face *face1) {
    if (adjacent(face0, face1))
        return;
    int t = omp_get_thread_num();
    Tensor b0, b1;
    bool is_ixn = intersection_midpoint(face0, face1, b0, b1);
    if (!is_ixn)
        return;
    Tensor n = normalize(old_pos(face0, b0) - old_pos(face1, b1));
    farthest_points(face0, face1, n, b0, b1);
    ::ixns[t].push_back(Ixn(face0, b0, face1, b1, n));
}

struct SeparationOpt: public NLConOpt {
    const vector<Ixn> &ixns;
    vector<Node*> nodes;
    Tensor inv_m;
    vector<double> tmp;
    SeparationOpt (const vector<Ixn> &ixns): ixns(ixns), inv_m(ZERO) {
        for (int i = 0; i < ixns.size(); i++) {
            if (is_free(ixns[i].f0))
                for (int v = 0; v < 3; v++)
                    include(ixns[i].f0->v[v]->node, nodes);
            if (is_free(ixns[i].f1))
                for (int v = 0; v < 3; v++)
                    include(ixns[i].f1->v[v]->node, nodes);
        }
        nvar = nodes.size()*3;
        ncon = ixns.size();
        for (int n = 0; n < nodes.size(); n++)
            inv_m = inv_m + 1/nodes[n]->a;
        tmp = vector<double>(nvar);
    }
    void initialize (double *x) const;
    double objective (const double *x) const;
    void obj_grad (const double *x, double *grad) const;
    double constraint (const double *x, int j, int &sign) const;
    void con_grad (const double *x, int j, double factor, double *grad) const;
    void finalize (const double *x);
    void precompute (const double *x) const;
};

void precompute_derivative(real_2d_array &a, real_2d_array &q, real_2d_array &r0, vector<double> &lambda,
                            real_1d_array &sm_1, vector<int> &legals, double **grads,
                            SeparationOpt &slx) {
    a.setlength(slx.nvar,legals.size());
    sm_1.setlength(slx.nvar);
    for (int i = 0; i < slx.nvar; ++i)
        sm_1[i] = 1.0/sqrt(slx.inv_m*slx.nodes[i/3]->a).item<double>();
    for (int k = 0; k < legals.size(); ++k)
        for (int i = 0; i < slx.nvar; ++i)
            a[i][k]=grads[legals[k]][i] * sm_1[i]; //sqrt(m^-1)
    real_1d_array tau, r1lam1, lamp;
    tau.setlength(slx.nvar);
    rmatrixqr(a, slx.nvar, legals.size(), tau);
    real_2d_array qtmp, r, r1;
    rmatrixqrunpackq(a, slx.nvar, legals.size(), tau, legals.size(), qtmp);
    rmatrixqrunpackr(a, slx.nvar, legals.size(), r);
    // get rid of degenerate G
    int newdim = 0;
    for (;newdim < legals.size(); ++newdim)
        if (abs(r[newdim][newdim]) < 1e-6)
            break;
    r0.setlength(newdim, newdim);
    r1.setlength(newdim, legals.size() - newdim);
    q.setlength(slx.nvar, newdim);
    for (int i = 0; i < slx.nvar; ++i)
        for (int j = 0; j < newdim; ++j)
            q[i][j] = qtmp[i][j];
    for (int i = 0; i < newdim; ++i) {
        for (int j = 0; j < newdim; ++j)
            r0[i][j] = r[i][j];
        for (int j = newdim; j < legals.size(); ++j)
            r1[i][j-newdim] = r[i][j];
    }
    r1lam1.setlength(newdim);
    for (int i = 0; i < newdim; ++i) {
        r1lam1[i] = 0;
        for (int j = newdim; j < legals.size(); ++j)
            r1lam1[i] += r1[i][j-newdim] * lambda[legals[j]];
    }
    ae_int_t info;
    alglib::densesolverreport rep;
    rmatrixsolve(r0, (ae_int_t)newdim, r1lam1, info, rep, lamp);
    for (int j = 0; j < newdim; ++j)
        lambda[legals[j]] += lamp[j];
    for (int j = newdim; j < legals.size(); ++j)
        lambda[legals[j]] = 0;
}

vector<Tensor> solve_ixns_forward(Tensor xold, Tensor bs, Tensor ns, vector<Ixn> &ixns) {
    auto slx = SeparationOpt(ixns);
    double x[slx.nvar];
    int sign;
    slx.initialize(x);
    auto lambda = augmented_lagrangian_method(slx);
//     // do qr decomposition on sqrt(m^-1)G^T
    vector<int> legals;
    double *grads[slx.ncon], tmp[slx.ncon];
    for (int i = 0; i < slx.ncon; ++i) {
        tmp[i] = slx.constraint(&slx.tmp[0],i,sign);
        if (sign==1 && tmp[i]>1e-6) continue;
        if (sign==-1 && tmp[i]<-1e-6) continue;
        grads[i] = new double[slx.nvar];
        for (int j = 0; j < slx.nvar; ++j)
            grads[i][j]=0;
        slx.con_grad(&slx.tmp[0],i,1,grads[i]);
        legals.push_back(i);
    }
    real_2d_array a, q, r;
    real_1d_array sm_1;//sqrt(m^-1)
    precompute_derivative(a, q, r, lambda, sm_1, legals, grads, slx);
    Tensor q_tn = arr2ten(q), r_tn = arr2ten(r);
    Tensor lam_tn = ptr2ten(&lambda[0], lambda.size());
    Tensor sm1_tn = ptr2ten(sm_1.getcontent(), sm_1.length());
    Tensor legals_tn = ptr2ten(&legals[0], legals.size());
    Tensor ans = ptr2ten(&slx.tmp[0], slx.nvar);
    return {ans.reshape({-1, 3}), q_tn, r_tn, lam_tn, sm1_tn, legals_tn};
}

void solve_ixns (vector<Ixn> &ixns) {
    auto slx = SeparationOpt(ixns);
    py::object func = py::module::import("separate_py").attr("solve_ixns");
    Tensor inp_xold, inp_b, inp_n;
    vector<Tensor> xolds(slx.nodes.size()), bs(slx.ncon*2), ns(slx.ncon);
    for (int i = 0; i < slx.nodes.size(); ++i)
        xolds[i] = ::xold[get_index(slx.nodes[i], *::meshes)];
    for (int j = 0; j < slx.ncon; ++j) {
        ns[j] = ixns[j].n;
        bs[j*2+0] = ixns[j].b0;
        bs[j*2+1] = ixns[j].b1;
    }
    inp_xold = torch::stack(xolds);
    inp_b = torch::stack(bs);
    inp_n = torch::stack(ns);
    Tensor out_x = func(inp_xold, inp_b, inp_n, ixns).cast<Tensor>();
    // Tensor out_x = solve_ixns_forward(inp_xold, inp_b, inp_n, ixns)[0];
    for (int i = 0; i < slx.nodes.size(); ++i)
        slx.nodes[i]->x = out_x[i];
}

vector<Tensor> compute_derivative(SeparationOpt &slx, real_1d_array &ans, vector<Ixn> &ixns,
                        real_2d_array &q, real_2d_array &r, real_1d_array &sm_1, vector<int> &legals, 
                        real_1d_array &dldx,
                        vector<double> &lambda, bool verbose=false) {
    real_1d_array qtx, dz, dlam0, dlam, ana, dldb, dldn0;
    ana.setlength(slx.nvar);
    qtx.setlength(q.cols());
    dz.setlength(slx.nvar);
    dldb.setlength(slx.ncon*6);
    dldn0.setlength(slx.ncon*3);
    dlam0.setlength(q.cols());
    dlam.setlength(slx.ncon);
    for (int i = 0; i < slx.nvar; ++i)
        ana[i] = dz[i] = 0;
    // qtx = qt * sqrt(m^-1) dldx
    for (int i = 0; i < q.cols(); ++i) {
        qtx[i] = 0;
        for (int j = 0; j < slx.nvar; ++j)
            qtx[i] += q[j][i] * dldx[j] * sm_1[j];
    }
    // dz = sqrt(m^-1) (sqrt(m^-1) dldx - q * qtx)
    for (int i = 0; i < slx.nvar; ++i) {
        dz[i] = dldx[i] * sm_1[i];
        for (int j = 0; j < q.cols(); ++j)
            dz[i] -= q[i][j] * qtx[j];
        dz[i] *= sm_1[i];
    }
    // dlam = R^-1 * qtx
    ae_int_t info;
    alglib::densesolverreport rep;
    rmatrixsolve(r, (ae_int_t)q.cols(), qtx, info, rep, dlam0);
// cout<<endl;
    for (int j = 0; j < slx.ncon; ++j)
        dlam[j] = 0;
    for (int k = 0; k < q.cols(); ++k)
        dlam[legals[k]] = dlam0[k];
    //part1: dldq * dqdxt = M dz
    for (int i = 0; i < slx.nvar; ++i)
        ana[i] += dz[i] / sm_1[i] / sm_1[i];
    //part2: dldg * dgdw * dwdxt
    for (int j = 0; j < slx.ncon; ++j) {
        if (lambda[j] == 0 && dlam[j] == 0)
            continue;
        Ixn &ixn=ixns[j];
        double *dldn = dldn0.getcontent()+j*3;
        double *dldb0 = dldb.getcontent()+j*6;
        double *dldb1 = dldb.getcontent()+j*6+3;
        for (int n = 0; n < 3; n++) {
            int i0 = find(ixn.f0->v[n]->node, slx.nodes);
            if (i0 != -1) {
                for (int k = 0; k < 3; ++k) {
                    dldb0[n] += -(dlam[j]*slx.tmp[i0*3+k]+lambda[j]*dz[i0*3+k])*ixn.n[k].item<double>();
    //part3: dldg * dgdn * dndxt
                    dldn[k] += -ixn.b0[n].item<double>()*(dlam[j]*slx.tmp[i0*3+k]+lambda[j]*dz[i0*3+k]);
                }
            } else {
    //part4: dldh * (dhdw + dhdn)
                for (int k = 0; k < 3; ++k) {
                    dldb0[n] += -(dlam[j] * ixn.n[k] * ixn.f0->v[n]->node->x[k]).item<double>();
                    dldn[k] += -(dlam[j] * ixn.b0[n] * ixn.f0->v[n]->node->x[k]).item<double>();
                }
            }
            int i1 = find(ixn.f1->v[n]->node, slx.nodes);
            if (i1 != -1) {
                for (int k = 0; k < 3; ++k) {
                    dldb1[n] -= -(dlam[j]*slx.tmp[i1*3+k]+lambda[j]*dz[i1*3+k])*ixn.n[k].item<double>();
    //part3: dldg * dgdn * dndxt
                    dldn[k] -= -ixn.b1[n].item<double>()*(dlam[j]*slx.tmp[i1*3+k]+lambda[j]*dz[i1*3+k]);
                }
            } else {
    //part4: dldh * (dhdw + dhdn)
                for (int k = 0; k < 3; ++k) {
                    dldb1[n] -= -(dlam[j] * ixn.n[k] * ixn.f1->v[n]->node->x[k]).item<double>();
                    dldn[k] -= -(dlam[j] * ixn.b1[n] * ixn.f1->v[n]->node->x[k]).item<double>();
                }
            }
        }
    }
    Tensor grad_xold = torch::from_blob(ana.getcontent(), {slx.nvar/3, 3}, TNOPT).clone();
    Tensor grad_b = torch::from_blob(dldb.getcontent(), {slx.ncon*2, 3}, TNOPT).clone();
    Tensor grad_n = torch::from_blob(dldn0.getcontent(), {slx.ncon, 3}, TNOPT).clone();
    return {grad_xold, grad_b, grad_n};
}

vector<Tensor> solve_ixns_backward(Tensor dldx_tn, Tensor ans_tn, Tensor q_tn, Tensor r_tn, Tensor lam_tn, Tensor sm1_tn, Tensor legals_tn, vector<Ixn> &ixns) {
    SeparationOpt slx = SeparationOpt(ixns);
    real_2d_array q = ten2arr(q_tn), r = ten2arr(r_tn);
    real_1d_array sm_1 = ten1arr(sm1_tn), ans = ten1arr(ans_tn), dldx = ten1arr(dldx_tn.reshape({-1}));
    vector<double> lambda = ten2vec<double>(lam_tn);
    vector<int> legals = ten2vec<int>(legals_tn);
    return compute_derivative(slx, ans, ixns, q, r, sm_1, legals, dldx, lambda);
}

void SeparationOpt::initialize (double *x) const {
    for (int n = 0; n < nodes.size(); n++)
        set_subvec(x, n, nodes[n]->x);
}

double SeparationOpt::objective (const double *x) const {
    double f = 0;
    for (int n = 0; n < nodes.size(); n++) {
        const Node *node = nodes[n];
        Tensor dx = get_subvec(x, n) - ::xold[get_index(node, *::meshes)];
        f += (inv_m*node->a*dot(dx,dx)/2).item<double>();
    }
    return f;
}

void SeparationOpt::obj_grad (const double *x, double *grad) const {
    for (int n = 0; n < nodes.size(); n++) {
        const Node *node = nodes[n];
        Tensor dx = get_subvec(x, n) - ::xold[get_index(node, *::meshes)];
        set_subvec(grad, n, inv_m*node->a*dx);
    }
}

double SeparationOpt::constraint (const double *x, int j, int &sign) const {
    const Ixn &ixn = ixns[j];
    sign = 1;
    double c = -::thickness.item<double>();
    for (int v = 0; v < 3; v++) {
        int i0 = find(ixn.f0->v[v]->node, nodes),
            i1 = find(ixn.f1->v[v]->node, nodes);
        Tensor x0 = (i0 != -1) ? get_subvec(x, i0) : ixn.f0->v[v]->node->x,
             x1 = (i1 != -1) ? get_subvec(x, i1) : ixn.f1->v[v]->node->x;
        c += (ixn.b0[v]*dot(ixn.n, x0)).item<double>();
        c -= (ixn.b1[v]*dot(ixn.n, x1)).item<double>();
    }
    return c;
}

void SeparationOpt::con_grad (const double *x, int j, double factor,
                                double *grad) const {
    const Ixn &ixn = ixns[j];
    for (int v = 0; v < 3; v++) {
        int i0 = find(ixn.f0->v[v]->node, nodes),
            i1 = find(ixn.f1->v[v]->node, nodes);
        if (i0 != -1)
            add_subvec(grad, i0, factor*ixn.b0[v]*ixn.n);
        if (i1 != -1)
            add_subvec(grad, i1, -factor*ixn.b1[v]*ixn.n);
    }
}

void SeparationOpt::precompute (const double *x) const {
    for (int n = 0; n < nodes.size(); n++)
        nodes[n]->x = get_subvec(x, n);
}

void SeparationOpt::finalize (const double *x) {
    precompute(x);
    for (int i = 0; i < nvar; ++i)
        tmp[i] = x[i];
}
