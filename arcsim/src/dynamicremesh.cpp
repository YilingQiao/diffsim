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

#include "dynamicremesh.hpp"
#include "geometry.hpp"
#include "magic.hpp"
#include "remesh.hpp"
#include "tensormax.hpp"
#include "timer.hpp"
#include "util.hpp"
#include <algorithm>
#include <cstdlib>
#include <map>
using namespace std;
using torch::Tensor;

static const bool verbose = false;

static Cloth::Remeshing *remeshing;
static bool plasticity;

void create_vert_sizing (Mesh &mesh, const vector<Plane> &planes);
void destroy_vert_sizing (Mesh &mesh);

// sizing field
struct Sizing {
    Tensor M;
    Sizing (): M(torch::zeros({2,2},TNOPT)) {}
};
Sizing &operator+= (Sizing &s1, const Sizing &s2) {
    s1.M = s1.M + s2.M; return s1;
}
Sizing operator+ (const Sizing &s1, const Sizing &s2) {
    Sizing s = s1; s += s2; return s;
}
Sizing &operator*= (Sizing &s, Tensor a) {
    s.M = s.M * a; return s;
}
Sizing operator* (const Sizing &s, Tensor a) {
    Sizing s2 = s; s2 *= a; return s2;
}
Sizing operator* (Tensor a, const Sizing &s) {
    return s*a;
}
Sizing &operator/= (Sizing &s, Tensor a) {
    s.M = s.M / a; return s;
}
Sizing operator/ (const Sizing &s, Tensor a) {
    Sizing s2 = s; s2 /= a; return s2;
}

Tensor norm2 (const Tensor &u, const Sizing &s) {
    return dot(u, matmul(s.M, u));
}
Tensor norm (const Tensor &u, const Sizing &s) {
    return sqrt(max(norm2(u,s), ZERO));
}
Tensor mean (const Sizing &s) {
    return s.M;
}

// The algorithm

bool fix_up_mesh (vector<Face*> &active, Mesh &mesh, vector<Edge*>* edges=0);

bool split_worst_edge (Mesh &mesh);

bool improve_some_face (vector<Face*> &active, Mesh &mesh);

void static_remesh (Cloth &cloth) {
    ::remeshing = &cloth.remeshing;
    Mesh &mesh = cloth.mesh;
    for (int v = 0; v < mesh.verts.size(); v++) {
        Sizing *sizing = new Sizing;
        sizing->M = torch::eye(2,TNOPT)*1.f/sq(remeshing->size_min);
        mesh.verts[v]->sizing = sizing;
    }
    while (split_worst_edge(mesh));
    vector<Face*> active = mesh.faces;
    while (improve_some_face(active, mesh));
    for (int v = 0; v < mesh.verts.size(); v++)
        delete mesh.verts[v]->sizing;
    update_indices(mesh);
    compute_ms_data(mesh);
    compute_masses(cloth);
}

void dynamic_remesh (Cloth &cloth, const vector<Plane> &planes,
                     bool plasticity) {
    ::remeshing = &cloth.remeshing;
    ::plasticity = plasticity;
    Mesh &mesh = cloth.mesh;
    create_vert_sizing(mesh, planes);
    vector<Face*> active = mesh.faces;
    fix_up_mesh(active, mesh);
    while (split_worst_edge(mesh));
    active = mesh.faces;
    while (improve_some_face(active, mesh));
    destroy_vert_sizing(mesh);
    update_indices(mesh);
    compute_ms_data(mesh);
    compute_masses(cloth);
}

// Sizing

Tensor angle (const Tensor &n1, const Tensor &n2) {
    return acos(clamp(dot(n1,n2),-1.,1.));
}

Tensor sqrt2 (const Tensor &A) {
    Eig eig = eigen_decomposition(A);
    int n = 2;
    for (int i = 0; i < n; i++)
        eig.l[i] = (eig.l[i]>=0).item<int>() ? sqrt(eig.l[i]) : -sqrt(-eig.l[i]);
    return matmul(eig.Q, matmul(diag(eig.l), eig.Q.t()));
}

Tensor pos (const Tensor &A) {
    Eig eig = eigen_decomposition(A);
    int n = 2;
    for (int i = 0; i < n; i++)
        eig.l[i] = max(eig.l[i], ZERO);
    return matmul(eig.Q, matmul(diag(eig.l), eig.Q.t()));
}

Tensor perp2 (const Tensor &A) {return torch::stack({A[1][1], -A[0][1], -A[1][0], A[0][0]}).reshape({2,2});}

Tensor compression_metric (const Tensor &e, const Tensor &S2, Tensor c) {
    Tensor D = e.t()*e - 4*sq(c)*perp2(S2)*::magic.rib_stiffening;
    return pos(-e + sqrt2(D))/(2*sq(c));
}

Tensor obstacle_metric (const Face *face, const vector<Plane> &planes) {
    Tensor o = torch::zeros({2,2},TNOPT);
    for (int v = 0; v < 3; v++) {
        Plane p = planes[face->v[v]->node->index];
        if ((norm2(p.second) == 0).item<int>())
            continue;
        Tensor h[3];
        for (int v1 = 0; v1 < 3; v1++)
            h[v1] = dot(face->v[v1]->node->x - p.first, p.second);
        Tensor dh = derivative(h[0], h[1], h[2], face);
        o = o + ger(dh,dh)/sq(h[v]);
    }
    return o/3.;
}

Sizing compute_face_sizing (const Face *face, const vector<Plane> &planes) {
    Sizing s;
    Tensor Sp = curvature<PS>(face);
    Tensor Sw1 = curvature<WS>(face);
    Tensor Sw2 = derivative(face->v[0]->node->n, face->v[1]->node->n,
                             face->v[2]->node->n, face);
    Tensor Mcurvp = !::plasticity ? torch::zeros({2,2},TNOPT)
                  : matmul(Sp.t(),Sp)/sq(remeshing->refine_angle);
    Tensor Mcurvw1 = matmul(Sw1.t(),Sw1)/sq(remeshing->refine_angle);
    Tensor Mcurvw2 = matmul(Sw2.t(),Sw2)/sq(remeshing->refine_angle);
    Tensor V = derivative(face->v[0]->node->v, face->v[1]->node->v,
                          face->v[2]->node->v, face);
    Tensor Mvel = matmul(V.t(),V)/sq(remeshing->refine_velocity);
    Tensor F = derivative(face->v[0]->node->x, face->v[1]->node->x,
                          face->v[2]->node->x, face);
    // Mat2x2 Mcomp = compression_metric(F.t()*F)
    //                / sq(remeshing->refine_compression);
    Tensor Mcomp = compression_metric(matmul(F.t(),F) - torch::eye(2,TNOPT), matmul(Sw2.t(),Sw2),
                                      remeshing->refine_compression);
    Tensor Mobs = (planes.empty()) ? torch::zeros({2,2},TNOPT) : obstacle_metric(face, planes);
    vector<Tensor> Ms(6);
    Ms[0] = Mcurvp;
    Ms[1] = Mcurvw1;
    Ms[2] = Mcurvw2;
    Ms[3] = Mvel;
    Ms[4] = Mcomp;
    Ms[5] = Mobs;
    s.M = ::magic.combine_tensors ? tensor_max(Ms)
        : Ms[0] + Ms[1] + Ms[2] + Ms[3] + Ms[4] + Ms[5];
    Eig eig = eigen_decomposition(s.M);
    for (int i = 0; i < 2; i++)
        eig.l[i] = clamp(eig.l[i],
                         1./sq(remeshing->size_max).item<double>(),
                         1./sq(remeshing->size_min).item<double>());
    Tensor lmax = max(eig.l[0], eig.l[1]);
    Tensor lmin = lmax*sq(remeshing->aspect_min);
    for (int i = 0; i < 2; i++)
        if ((eig.l[i] < lmin).item<int>())
            eig.l[i] = lmin;
    s.M = matmul(eig.Q,matmul(diag(eig.l),eig.Q.t()));
    return s;
}

static Tensor area (const Tensor &u0, const Tensor &u1, const Tensor &u2) {
    return 0.5*wedge(u1-u0, u2-u0);
}

static Tensor area (const Face *face) {
    return area(face->v[0]->u, face->v[1]->u, face->v[2]->u);
}

static Tensor perimeter (const Tensor &u0, const Tensor &u1, const Tensor &u2) {
    return norm(u0 - u1) + norm(u1 - u2) + norm(u2 - u0);
}

static Tensor aspect (const Tensor &u0, const Tensor &u1, const Tensor &u2) {
    Tensor a = area(u0, u1, u2);
    Tensor p = perimeter(u0, u1, u2);
    return 12*sqrt(3)*a/sq(p);
}

static Tensor aspect (const Face *face) {
    return aspect(face->v[0]->u, face->v[1]->u, face->v[2]->u);
}

Sizing compute_vert_sizing (const Vert *vert,
                            const map<Face*,Sizing> &face_sizing) {
    Sizing sizing;
    for (int f = 0; f < vert->adjf.size(); f++) {
        const Face *face = vert->adjf[f];
        sizing += face->a/3. * face_sizing.find((Face*)face)->second;
    }
    sizing /= vert->a;
    return sizing;
}

// Cache

void create_vert_sizing (Mesh &mesh, const vector<Plane> &planes) {
    map<Face*,Sizing> face_sizing;
    for (int f = 0; f < mesh.faces.size(); f++)
        face_sizing[mesh.faces[f]] = compute_face_sizing(mesh.faces[f], planes);
    for (int v = 0; v < mesh.verts.size(); v++)
        mesh.verts[v]->sizing =
            new Sizing(compute_vert_sizing(mesh.verts[v], face_sizing));
}

void destroy_vert_sizing (Mesh &mesh) {
    for (int v = 0; v < mesh.verts.size(); v++)
        delete mesh.verts[v]->sizing;
}

Tensor edge_metric (const Vert *vert0, const Vert *vert1) {
    if (!vert0 || !vert1)
        return ZERO;
    Tensor du = vert0->u - vert1->u;
    return sqrt((norm2(du, *vert0->sizing) + norm2(du, *vert1->sizing))/2.);
}

Tensor edge_metric (const Edge *edge) {
    Tensor m = (edge_metric(edge_vert(edge,0,0), edge_vert(edge,0,1))
              + edge_metric(edge_vert(edge,1,0), edge_vert(edge,1,1)));
    return (edge->adjf[0] && edge->adjf[1]) ? m/2 : m;
}

// Helpers

template <typename T> void include_all (const vector<T> &u, vector<T> &v) {for (int i = 0; i < u.size(); i++) include(u[i], v);}
template <typename T> void exclude_all (const vector<T> &u, vector<T> &v) {for (int i = 0; i < u.size(); i++) exclude(u[i], v);}
template <typename T> void set_null_all (const vector<T> &u, vector<T> &v) {for (int i = 0; i < u.size(); i++) exclude(u[i], v);}

void update_active (const RemeshOp &op, vector<Face*> &active) {
    exclude_all(op.removed_faces, active);
    include_all(op.added_faces, active);
}

void update_active (const vector<RemeshOp> &ops, vector<Face*> &active) {
    for (int i = 0; i < ops.size(); i++)
        update_active(ops[i], active);
}

// Fixing-upping

RemeshOp flip_edges (vector<Face*> &active, Mesh &mesh);

Vert *most_valent_vert (const vector<Face*> &faces);

// Vert *farthest_neighbor (const Vert *vert);

bool fix_up_mesh (vector<Face*> &active, Mesh &mesh, vector<Edge*> *edges) {
    RemeshOp flip_ops = flip_edges(active, mesh);
    update_active(flip_ops, active);
    if (edges)
        set_null_all(flip_ops.removed_edges, *edges);
    flip_ops.done();
    return !flip_ops.empty();
}

RemeshOp flip_some_edges (vector<Face*> &active, Mesh &mesh);

RemeshOp flip_edges (vector<Face*> &active, Mesh &mesh) {
    RemeshOp ops;
    for (int i = 0; i < 3*mesh.verts.size(); i++) {// don't loop without bound
        RemeshOp op = flip_some_edges(active, mesh);
        if (op.empty())
            break;
        ops = compose(ops, op);
    }
    return ops;
}

vector<Edge*> find_edges_to_flip (const vector<Face*> &active);
vector<Edge*> independent_edges (const vector<Edge*> &edges);

bool inverted (const Face *face) {return (area(face) < 1e-12).item<int>();}
bool degenerate (const Face *face) {
    return (aspect(face) < remeshing->aspect_min/4).item<int>();}
bool any_inverted (const vector<Face*> faces) {
    for (int i=0; i<faces.size(); i++) if (inverted(faces[i])) return true;
    return false;
}
bool any_degenerate (const vector<Face*> faces) {
    for (int i=0; i<faces.size(); i++) if (degenerate(faces[i])) return true;
    return false;
}

RemeshOp flip_some_edges (vector<Face*> &active, Mesh &mesh) {
    RemeshOp ops;
    static int n_edges_prev = 0;
    vector<Edge*> edges = independent_edges(find_edges_to_flip(active));
    if (edges.size() == n_edges_prev) // probably infinite loop
        return ops;
    n_edges_prev = edges.size();
    for (int e = 0; e < edges.size(); e++) {
        Edge *edge = edges[e];
        RemeshOp op = flip_edge(edge);
        op.apply(mesh);
        if (any_inverted(op.added_faces)) {
            op.inverse().apply(mesh);
            op.inverse().done();
            continue;
        }
        update_active(op, active);
        ops = compose(ops, op);
    }
    return ops;
}

bool should_flip (const Edge *edge);

vector<Edge*> find_edges_to_flip (const vector<Face*> &active){
    vector<Edge*> edges;
    for (int f = 0; f < active.size(); f++) {
        include(active[f]->adje[0], edges);
        include(active[f]->adje[1], edges);
        include(active[f]->adje[2], edges);
    }
    vector<Edge*> fedges;
    for (int e = 0; e < edges.size(); e++) {
        Edge *edge = edges[e];
        if (is_seam_or_boundary(edge) || edge->label != 0
            || !should_flip(edge))
            continue;
        fedges.push_back(edge);
    }
    return fedges;
}

bool independent (const Edge *edge, const vector<Edge*> &edges) {
    for (int i = 0; i < edges.size(); i++) {
        Edge *edge1 = edges[i];
        if (edge->n[0] == edge1->n[0] || edge->n[0] == edge1->n[1]
            || edge->n[1] == edge1->n[0] || edge->n[1] == edge1->n[1])
            return false;
    }
    return true;
}

vector<Edge*> independent_edges (const vector<Edge*> &edges) {
    vector<Edge*> iedges;
    for (int e = 0; e < edges.size(); e++)
        if (independent(edges[e], iedges))
            iedges.push_back(edges[e]);
    return iedges;
}

// from Bossen and Heckbert 1996
bool should_flip (const Edge *edge) {
    const Vert *vert0 = edge_vert(edge, 0, 0), *vert1 = edge_vert(edge, 0, 1),
               *vert2 = edge_opp_vert(edge, 0), *vert3 = edge_opp_vert(edge, 1);
    Tensor x = vert0->u, z = vert1->u, w = vert2->u, y = vert3->u;
    Tensor M = (mean(*vert0->sizing) + mean(*vert1->sizing)
                + mean(*vert2->sizing) + mean(*vert3->sizing))/4.;
    return (wedge(z-y, x-y)*dot(x-w, matmul(M,(z-w))) + dot(z-y, matmul(M,(x-y)))*wedge(x-w, z-w)
           < -::magic.edge_flip_threshold*(wedge(z-y, x-y) + wedge(x-w, z-w))).item<int>();
}

// Splitting

vector<Edge*> find_bad_edges (const Mesh &mesh);

Sizing mean_vert_sizing (const Vert *vert0, const Vert *vert1);

Vert *adjacent_vert (const Node *node, const Vert *vert);

bool split_worst_edge (Mesh &mesh) {
    vector<Edge*> edges = find_bad_edges(mesh);
    for (int e = 0; e < edges.size(); e++) {
        Edge *edge = edges[e];
        if (!edge) continue;
        Node *node0 = edge->n[0], *node1 = edge->n[1];
        RemeshOp op = split_edge(edge);
        op.apply(mesh);
        for (int v = 0; v < op.added_verts.size(); v++) {
            Vert *vertnew = op.added_verts[v];
            Vert *v0 = adjacent_vert(node0, vertnew),
                 *v1 = adjacent_vert(node1, vertnew);
            vertnew->sizing = new Sizing(mean_vert_sizing(v0, v1));
        }
        set_null_all(op.removed_edges, edges);
        op.done();
        if (verbose)
            cout << "Split " << node0 << " and " << node1 << endl;
        vector<Face*> active = op.added_faces;
        fix_up_mesh(active, mesh, &edges);
    }
    return !edges.empty();
}

// don't use edge pointer as secondary sort key, otherwise not reproducible
struct Deterministic_sort {
    inline bool operator()(const std::pair<double,Edge*> &left, const std::pair<double,Edge*> &right) {
        return left.first < right.first;
    }
} deterministic_sort;

vector<Edge*> find_bad_edges (const Mesh &mesh) {
    vector< pair<double,Edge*> > edgems;
    for (int e = 0; e < mesh.edges.size(); e++) {
        Edge *edge = mesh.edges[e];
        double m = edge_metric(edge).item<double>();
        if (m > 1)
            edgems.push_back(make_pair(m, edge));
    }
    sort(edgems.begin(), edgems.end(), deterministic_sort);
    vector<Edge*> edges(edgems.size());
    for (int e = 0; e < edgems.size(); e++)
        edges[e] = edgems[edgems.size()-e-1].second;
    return edges;
}

Sizing mean_vert_sizing (const Vert *vert0, const Vert *vert1) {
    Sizing sizing = *vert0->sizing;
    sizing += *vert1->sizing;
    return sizing/(ONE*2.);
}

Vert *adjacent_vert (const Node *node, const Vert *vert) {
    const Edge *edge = get_edge(node, vert->node);
    for (int i = 0; i < 2; i++)
        for (int s = 0; s < 2; s++)
            if (edge_vert(edge, s, i) == vert)
                return edge_vert(edge, s, 1-i);
    return NULL;
}

// Collapsing

vector<int> sort_edges_by_length (const Face *face);

RemeshOp try_edge_collapse (Edge *edge, int which, Mesh &mesh);

bool improve_some_face (vector<Face*> &active, Mesh &mesh) {
    for (int f = 0; f < active.size(); f++) {
        Face *face = active[f];
        for (int e = 0; e < 3; e++) {
            Edge *edge = face->adje[e];
            RemeshOp op;
            if (op.empty()) op = try_edge_collapse(edge, 0, mesh);
            if (op.empty()) op = try_edge_collapse(edge, 1, mesh);
            if (op.empty()) continue;
            op.done();
            update_active(op, active);
            vector<Face*> fix_active = op.added_faces;
            RemeshOp flip_ops = flip_edges(fix_active, mesh);
            update_active(flip_ops, active);
            flip_ops.done();
            return true;
        }
        remove(f--, active);
    }
    return false;
}

bool has_labeled_edges (const Node *node);
bool can_collapse (const Edge *edge, int which);
bool any_nearly_invalid (const vector<Edge*> edges) {
    for (int i = 0; i < edges.size(); i++)
        if ((edge_metric(edges[i]) > 0.9).item<int>()) return true;
    return false;
}

RemeshOp try_edge_collapse (Edge *edge, int which, Mesh &mesh) {
    Node *node0 = edge->n[which], *node1 = edge->n[1-which];
    if (node0->preserve
        || (is_seam_or_boundary(node0) && !is_seam_or_boundary(edge))
        || (has_labeled_edges(node0) && !edge->label))
        return RemeshOp();
    if (!can_collapse(edge, which))
        return RemeshOp();
    RemeshOp op = collapse_edge(edge, which);
    op.apply(mesh);
    if (op.empty())
        return op;
    // if (any_inverted(op.added_faces) || any_degenerate(op.added_faces)
    //     || any_nearly_invalid(op.added_edges)) {
    //     op.inverse().apply(mesh);
    //     op.inverse().done();
    //     return RemeshOp();
    // }
    for (int v = 0; v < op.removed_verts.size(); v++)
        delete op.removed_verts[v]->sizing;
    // delete op.removed_nodes[0]->res;
    if (verbose)
        cout << "Collapsed " << node0 << " into " << node1 << endl;
    return op;
}

bool has_labeled_edges (const Node *node) {
    for (int e = 0; e < node->adje.size(); e++)
        if (node->adje[e]->label)
            return true;
    return false;
}

bool can_collapse (const Edge *edge, int i) {
    for (int s = 0; s < 2; s++) {
        const Vert *vert0 = edge_vert(edge,s,i), *vert1 = edge_vert(edge,s,1-i);
        if (!vert0 || (s == 1 && vert0 == edge_vert(edge, 0, i)))
            continue;
        for (int f = 0; f < vert0->adjf.size(); f++) {
            const Face *face = vert0->adjf[f];
            if (is_in(vert1, face->v))
                continue;
            const Vert *vs[3] = {face->v[0], face->v[1], face->v[2]};
            replace(vert0, vert1, vs);
            Tensor a = wedge(vs[1]->u - vs[0]->u, vs[2]->u - vs[0]->u)/2;
            Tensor asp = aspect(vs[0]->u, vs[1]->u, vs[2]->u);
            if ((a < 1e-6).item<int>() || (asp < remeshing->aspect_min).item<int>())
                return false;
            for (int e = 0; e < 3; e++)
                if (vs[e]!=vert1 && (edge_metric(vs[NEXT(e)], vs[PREV(e)]) > 0.9).item<int>())
                    return false;
        }
    }
    return true;
}
