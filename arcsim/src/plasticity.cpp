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

#include "plasticity.hpp"

#include "bah.hpp"
#include "geometry.hpp"
#include "optimization.hpp"
#include "physics.hpp"
#include <omp.h>

using namespace std;
using torch::Tensor;

static const double mu = 1e-6;

Tensor edges_to_face (const Tensor &theta, const Face *face);
Tensor face_to_edges (const Tensor &S, const Face *face);

void reset_plasticity (Cloth &cloth) {
    Mesh &mesh = cloth.mesh;
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->y = mesh.nodes[n]->x;
    for (int e = 0; e < mesh.edges.size(); e++) {
        Edge *edge = mesh.edges[e];
        edge->theta_ideal = edge->reference_angle = edge->theta;
        edge->damage = ZERO;
    }
    for (int f = 0; f < mesh.faces.size(); f++) {
        Face *face = mesh.faces[f];
        Tensor theta = torch::stack({face->adje[0]->theta,
                          face->adje[1]->theta,
                          face->adje[2]->theta});
        face->S_plastic = edges_to_face(theta, face);
        face->damage = ZERO;
    }
}

void recompute_edge_plasticity (Mesh &mesh);

void optimize_plastic_embedding (Cloth &cloth);

void plastic_update (Cloth &cloth) {
    Mesh &mesh = cloth.mesh;
    const vector<Cloth::Material*> &materials = cloth.materials;
    for (int f = 0; f < mesh.faces.size(); f++) {
        Face *face = mesh.faces[f];
        Tensor S_yield = materials[face->label]->yield_curv;
        Tensor theta = torch::stack({face->adje[0]->theta,
                          face->adje[1]->theta,
                          face->adje[2]->theta});
        Tensor S_total = edges_to_face(theta, face);
        Tensor S_elastic = S_total - face->S_plastic;
        Tensor dS = frobenius_norm(S_elastic);
        if ((dS > S_yield).item<int>()) {
            face->S_plastic = face->S_plastic + S_elastic/dS*(dS - S_yield);
            face->damage = face->damage + dS/S_yield - 1;
        }
    }
    recompute_edge_plasticity(cloth.mesh);
}

// ------------------------------------------------------------------ //

struct EmbedOpt: public NLOpt {
    Cloth &cloth;
    Mesh &mesh;
    Tensor y0;
    mutable Tensor f;
    mutable SpMat J;
    EmbedOpt (Cloth &cloth): cloth(cloth), mesh(cloth.mesh) {
        int nn = mesh.nodes.size();
        nvar = nn*3;
        y0 = torch::zeros({nn,3},TNOPT);
        for (int n = 0; n < nn; n++)
            y0[n] = mesh.nodes[n]->y;
        // f.resize(nv);
        // J = SpMat<Mat3x3>(nv,nv);
    }
    void initialize (Tensor &x) const;
    void precompute (const Tensor &x) const;
    Tensor objective (const Tensor &x) const;
    void gradient (const Tensor &x, Tensor &g) const;
    bool hessian (const Tensor &x, SpMat &H) const;
    void finalize (const Tensor &x) const;
};

void reduce_stretching_stiffnesses (vector<Cloth::Material*> &materials);
void restore_stretching_stiffnesses (vector<Cloth::Material*> &materials);

void optimize_plastic_embedding (Cloth &cloth) {
    // vector<Cloth::Material> materials = cloth.materials;
    reduce_stretching_stiffnesses(cloth.materials);
    line_search_newtons_method(EmbedOpt(cloth), OptOptions().max_iter(1));
    restore_stretching_stiffnesses(cloth.materials);
    // cloth.materials = materials;
}

void EmbedOpt::initialize (Tensor &x) const {
    for (int n = 0; n < mesh.nodes.size(); n++)
        set_subvec(x, n, ZERO3);
}

void EmbedOpt::precompute (const Tensor &x) const {
    int nn = mesh.nodes.size();
    f = torch::zeros({nn,3}, TNOPT);
    J = SpMat(nn,nn);// torch::zeros({nn*3,nn*3},TNOPT);
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->y = y0[n] + get_subvec(x, n);
    add_internal_forces<PS>(cloth, J, f, ZERO);
}

Tensor EmbedOpt::objective (const Tensor &x) const {
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->y = y0[n] + get_subvec(x, n);
    return internal_energy<PS>(cloth);
}

void EmbedOpt::gradient (const Tensor &x, Tensor &g) const {
    for (int n = 0; n < mesh.nodes.size(); n++) {
        const Node *node = mesh.nodes[n];
        set_subvec(g, n, -f[n]);
    }
}

bool EmbedOpt::hessian (const Tensor &x, SpMat &H) const {
    // H = J + torch::eye(nvar,TNOPT)*::mu;
    for (int i = 0; i < mesh.nodes.size(); i++) {
        const SpVec &Ji = J.rows[i];
        for (int jj = 0; jj < Ji.indices.size(); jj++) {
            int j = Ji.indices[jj];
            const Tensor &Jij = Ji.entries[jj];
            H(i,j) = Jij;
        }
        add_submat(H, i, i, EYE3*::mu);
    }
    return true;
}

void EmbedOpt::finalize (const Tensor &x) const {
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->y = y0[n] + get_subvec(x, n);
}

void reduce_stretching_stiffnesses (vector<Cloth::Material*> &materials) {
    for (int m = 0; m < materials.size(); m++)
        materials[m]->stretching = materials[m]->stretching * 1e-2;
}

void restore_stretching_stiffnesses (vector<Cloth::Material*> &materials) {
    for (int m = 0; m < materials.size(); m++)
        materials[m]->stretching = materials[m]->stretching * 1e2;
}

// ------------------------------------------------------------------ //

Tensor edges_to_face (const Tensor &theta, const Face *face) {
    Tensor S = torch::zeros({2,2},TNOPT);
    for (int e = 0; e < 3; e++) {
        const Edge *edge = face->adje[e];
        Tensor e_mat = face->v[PREV(e)]->u - face->v[NEXT(e)]->u,
             t_mat = perp(normalize(e_mat));
        S = S - 1/2.*theta[e]*norm(e_mat)*ger(t_mat, t_mat);
    }
    S = S / face->a;
    return S;
}

Tensor face_to_edges (const Tensor &S, const Face *face) {
    Tensor s = face->a*torch::stack({S[0][0], S[1][1], S[0][1]});
    Tensor A = torch::zeros({3,3},TNOPT);
    for (int e = 0; e < 3; e++) {
        const Edge *edge = face->adje[e];
        Tensor e_mat = face->v[PREV(e)]->u - face->v[NEXT(e)]->u,
             t_mat = perp(normalize(e_mat));
        Tensor Se = -1/2.*norm(e_mat)*ger(t_mat, t_mat);
        A[0][e] = Se[0][0];
        A[1][e] = Se[1][1];
        A[2][e] = Se[0][1];
    }
    return matmul(A.inverse(),s);
}

void recompute_edge_plasticity (Mesh &mesh) {
    for (int e = 0; e < mesh.edges.size(); e++) {
        mesh.edges[e]->theta_ideal = ZERO;
        mesh.edges[e]->damage = ZERO;
    }
    for (int f = 0; f < mesh.faces.size(); f++) {
        const Face *face = mesh.faces[f];
        Tensor theta = face_to_edges(face->S_plastic, face);
        for (int e = 0; e < 3; e++) {
            face->adje[e]->theta_ideal = face->adje[e]->theta_ideal + theta[e];
            face->adje[e]->damage = face->adje[e]->damage + face->damage;
        }
    }
    for (int e = 0; e < mesh.edges.size(); e++) {
        Edge *edge = mesh.edges[e];
        if (edge->adjf[0] && edge->adjf[1]) {// edge has two adjacent faces
            edge->theta_ideal = edge->theta_ideal / 2;
            edge->damage = edge->damage / 2;
        }
        edge->reference_angle = edge->theta_ideal;
    }
}

// ------------------------------------------------------------------ //

const vector<Residual> *res_old;

void resample_callback (Face *face_new, const Face *face_old);

vector<Residual> back_up_residuals (Mesh &mesh) {
    vector<Residual> res(mesh.faces.size());
    for (int f = 0; f < mesh.faces.size(); f++) {
        const Face *face = mesh.faces[f];
        Tensor theta = ZERO3;
        for (int e = 0; e < 3; e++) {
            const Edge *edge = face->adje[e];
            theta[e] = edge->theta_ideal - dihedral_angle<PS>(edge);
        }
        res[f].S_res = edges_to_face(theta, face);
        res[f].damage = face->damage;
    }
    return res;
}

void restore_residuals (Mesh &mesh, const Mesh &old_mesh,
                        const vector<Residual> &res_old) {
    ::res_old = &res_old;
    BahNode *tree = new_bah_tree(old_mesh);
#pragma omp parallel for
    for (int f = 0; f < mesh.faces.size(); f++) {
        Face *face = mesh.faces[f];
        Tensor theta = ZERO3;
        for (int e = 0; e < 3; e++)
            theta[e] = dihedral_angle<PS>(face->adje[e]);
        face->S_plastic = edges_to_face(theta, face);
        face->damage = ZERO;
        for_overlapping_faces(face, tree, resample_callback);
    }
    delete_bah_tree(tree);
    recompute_edge_plasticity(mesh);
}

Tensor overlap_area (const Face *face0, const Face *face1);

void resample_callback (Face *face_new, const Face *face_old) {
    Tensor a = overlap_area(face_new, face_old)/face_new->a;
    const Residual &res = (*::res_old)[face_old->index];
    face_new->S_plastic = face_new->S_plastic + a*res.S_res;
    face_new->damage = face_new->damage + a*res.damage;
}

// ------------------------------------------------------------------ //

vector<Tensor> sutherland_hodgman (const vector<Tensor> &poly0,
                                 const vector<Tensor> &poly1);
Tensor area (const vector<Tensor> &poly);

Tensor overlap_area (const Face *face0, const Face *face1) {
    vector<Tensor> u0(3), u1(3);
    Tensor u0min(face0->v[0]->u), u0max(u0min),
         u1min(face1->v[0]->u), u1max(u1min);
    for (int i = 0; i < 3; i++) {
        u0[i] = face0->v[i]->u;
        u1[i] = face1->v[i]->u;
        u0min = min(u0min, u0[i]);
        u0max = max(u0max, u0[i]);
        u1min = min(u1min, u1[i]);
        u1max = max(u1max, u1[i]);
    }
    if ((u0min[0] > u1max[0]).item<int>() || (u0max[0] < u1min[0]).item<int>()
        || (u0min[1] > u1max[1]).item<int>() || (u0max[1] < u1min[1]).item<int>()) {
        return ZERO;
    }
    return area(sutherland_hodgman(u0, u1));
}

vector<Tensor> clip (const vector<Tensor> &poly, const Tensor &clip0,
                                             const Tensor &clip1);

vector<Tensor> sutherland_hodgman (const vector<Tensor> &poly0,
                                 const vector<Tensor> &poly1) {
    vector<Tensor> out(poly0);
    for (int i = 0; i < 3; i++)
        out = clip(out, poly1[i], poly1[(i+1)%poly1.size()]);
    return out;
}

Tensor distance_p (const Tensor &v, const Tensor &v0, const Tensor &v1) {
    return wedge(v1-v0, v-v0);}
Tensor lerpp (Tensor t, const Tensor &a, const Tensor &b) {return a + t*(b-a);}

vector<Tensor> clip (const vector<Tensor> &poly, const Tensor &clip0,
                                             const Tensor &clip1) {
    if (poly.empty())
        return poly;
    vector<Tensor> newpoly;
    for (int i = 0; i < poly.size(); i++) {
        const Tensor &v0 = poly[i], &v1 = poly[(i+1)%poly.size()];
        Tensor d0 = distance_p(v0, clip0, clip1), d1 = distance_p(v1, clip0, clip1);
        if ((d0 >= 0).item<int>())
            newpoly.push_back(v0);
        if (!(((d0<0).item<int>() && (d1<0).item<int>()) || ((d0==0).item<int>() && (d1==0).item<int>()) || ((d0>0).item<int>() && (d1>0).item<int>())))
            newpoly.push_back(lerpp(d0/(d0-d1), v0, v1));
    }
    return newpoly;
}

Tensor area (const vector<Tensor> &poly) {
    if (poly.empty())
        return ZERO;
    Tensor a = ZERO;
    for (int i = 1; i < poly.size()-1; i++)
        a = a + wedge(poly[i]-poly[0], poly[i+1]-poly[0])/2;
    return a;
}
