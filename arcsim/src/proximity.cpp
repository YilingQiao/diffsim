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

#include "proximity.hpp"

#include "collisionutil.hpp"
#include "geometry.hpp"
#include "magic.hpp"
#include "simulation.hpp"
#include <vector>
#include <algorithm>
#include <utility>
#include <omp.h>
using namespace std;
using torch::Tensor;

template <typename T> struct Min {
    Tensor key;
    T val;
    Min (): key(infinity), val() {}
    void add (Tensor key, T val) {
        if ((key < this->key).template item<int>()) {
            this->key = key;
            this->val = val;
        }
    }
};

static vector< Min<Face*> > node_prox[2];
static vector< Min<Edge*> > edge_prox[2];
static vector< Min<Node*> > face_prox[2];
static vector< Min<Face*> > obs_node_prox[2];
static vector< Min<Edge*> > obs_edge_prox[2];
static vector< Min<Node*> > obs_face_prox[2];
static vector<pair<Face const*, Face const*> > *prox_faces = NULL;

void find_proximities (const Face *face0, const Face *face1);
Constraint *make_constraint (const Node *node, const Face *face,
                             Tensor mu, Tensor mu_obs);
Constraint *make_constraint (const Edge *edge0, const Edge *edge1,
                             Tensor mu, Tensor mu_obs);

void add_proximity (const Node *node, const Face *face, double thick);
void add_proximity (const Edge *edge0, const Edge *edge1, double thick);

void obs_add_proximity (const Node *node, const Face *face, double thick);
void obs_add_proximity (const Edge *edge0, const Edge *edge1, double thick);

void compute_proximities(const Face *face0, const Face *face1) {
    BOX nb[6], eb[6], fb[2];
    for (int v = 0; v < 3; ++v) {
        nb[v] = node_box(face0->v[v]->node, false);
        nb[v+3] = node_box(face1->v[v]->node, false);
    }
    for (int v = 0; v < 3; ++v) {
        eb[v] = nb[NEXT(v)]+nb[PREV(v)];//edge_box(face0->adje[v], true);//
        eb[v+3] = nb[NEXT(v)+3]+nb[PREV(v)+3];//edge_box(face1->adje[v], true);//
    }
    fb[0] = nb[0]+nb[1]+nb[2];
    fb[1] = nb[3]+nb[4]+nb[5];
    double thick = 2*::magic.repulsion_thickness.item<double>();

    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v], fb[1], thick))
            continue;
        add_proximity(face0->v[v]->node, face1, thick);
        obs_add_proximity(face0->v[v]->node, face1, thick);
    }
    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v+3], fb[0], thick))
            continue;
        add_proximity(face1->v[v]->node, face0, thick);
        obs_add_proximity(face1->v[v]->node, face0, thick);
    }
    for (int e0 = 0; e0 < 3; e0++)
        for (int e1 = 0; e1 < 3; e1++) {
            if (!overlap(eb[e0], eb[e1+3], thick))
                continue;
            add_proximity(face0->adje[e0], face1->adje[e1], thick);
            obs_add_proximity(face0->adje[e0], face1->adje[e1], thick);
        }
}

vector<Constraint*> proximity_constraints (const vector<Mesh*> &meshes,
                                           const vector<Mesh*> &obs_meshes,
                                           Tensor mu, Tensor mu_obs) {
    vector<Constraint*> cons;
    int nthreads = omp_get_max_threads();
    if (!prox_faces)
      prox_faces = new vector<pair<Face const*, Face const*> >[nthreads];
    for (int t = 0; t < nthreads; ++t)
      prox_faces[t].clear();
    ::meshes = &meshes;
    ::obs_meshes = &obs_meshes;
    const Tensor dmin = 2*::magic.repulsion_thickness;
    vector<AccelStruct*> accs = create_accel_structs(meshes, false),
                         obs_accs = create_accel_structs(obs_meshes, false);
    int nn = size<Node>(meshes),
        ne = size<Edge>(meshes),
        nf = size<Face>(meshes);
    int obs_nn = size<Node>(obs_meshes),
        obs_ne = size<Edge>(obs_meshes),
        obs_nf = size<Face>(obs_meshes);
    for (int i = 0; i < 2; i++) {
        ::node_prox[i].assign(nn, Min<Face*>());
        ::edge_prox[i].assign(ne, Min<Edge*>());
        ::face_prox[i].assign(nf, Min<Node*>());
        ::obs_node_prox[i].assign(obs_nn, Min<Face*>());
        ::obs_edge_prox[i].assign(obs_ne, Min<Edge*>());
        ::obs_face_prox[i].assign(obs_nf, Min<Node*>());
    }
    for_overlapping_faces(accs, obs_accs, dmin, find_proximities);
    vector<pair<Face const*, Face const*> > tot_faces;
    for (int t = 0; t < nthreads; ++t)
        append(tot_faces, ::prox_faces[t]);
    random_shuffle(tot_faces.begin(), tot_faces.end());
    #pragma omp parallel for
    for (int i = 0; i < tot_faces.size(); ++i)
        compute_proximities(tot_faces[i].first,tot_faces[i].second);
    for (int n = 0; n < nn; n++)
        for (int i = 0; i < 2; i++) {
            Min<Face*> &m = ::node_prox[i][n];
            if ((m.key < dmin).item<int>())
                cons.push_back(make_constraint(get<Node>(n, meshes), m.val,
                                               mu, mu_obs));
        }
    for (int e = 0; e < ne; e++)
        for (int i = 0; i < 2; i++) {
            Min<Edge*> &m = ::edge_prox[i][e];
            if ((m.key < dmin).item<int>())
                cons.push_back(make_constraint(get<Edge>(e, meshes), m.val,
                                               mu, mu_obs));
        }
    for (int f = 0; f < nf; f++)
        for (int i = 0; i < 2; i++) {
            Min<Node*> &m = ::face_prox[i][f];
            if ((m.key < dmin).item<int>())
                cons.push_back(make_constraint(m.val, get<Face>(f, meshes),
                                               mu, mu_obs));
        }

    //rigid body
    // cout << "make_constraint " << ne << " " << nn << " " << nf << endl;
    for (int n = 0; n < obs_nn; n++) {
        for (int i = 0; i < 2; i++) {
            Min<Face*> &m = ::obs_node_prox[i][n];
            if ((m.key < dmin).item<int>())
                cons.push_back(make_constraint(get<Node>(n, obs_meshes), m.val,
                                               mu, mu_obs));
        }
    }
    for (int e = 0; e < obs_ne; e++) {
        for (int i = 0; i < 2; i++) {
            Min<Edge*> &m = ::obs_edge_prox[i][e];
            if ((m.key < dmin).item<int>())
                cons.push_back(make_constraint(get<Edge>(e, obs_meshes), m.val,
                                               mu, mu_obs));
        }
    }
    for (int f = 0; f < obs_nf; f++) {
        for (int i = 0; i < 2; i++) {
            Min<Node*> &m = ::obs_face_prox[i][f];
            if ((m.key < dmin).item<int>())
                cons.push_back(make_constraint(m.val, get<Face>(f, obs_meshes),
                                               mu, mu_obs));
        }
    }


// cout <<"contraints= " << cons.size() << endl;
    destroy_accel_structs(accs);
    destroy_accel_structs(obs_accs);
    return cons;
}

void find_proximities (const Face *face0, const Face *face1) {
    int t = omp_get_thread_num();
    prox_faces[t].push_back(make_pair(face0, face1));
}

void add_proximity (const Node *node, const Face *face, double thick) {
    if (node == face->v[0]->node
     || node == face->v[1]->node
     || node == face->v[2]->node)
        return;
    //if (!overlap(node_box(node, true), face_box(face, true), 2*::magic.repulsion_thickness.item<double>()))
    //    return;
    Tensor n;
    Tensor w[4];
    w[0]=w[1]=w[2]=w[3]=ZERO;
    n=ZERO3;
    bool over = false;
    Tensor d = signed_vf_distance(node->x, face->v[0]->node->x,
                                  face->v[1]->node->x, face->v[2]->node->x,
                                  &n, w, thick, over);
    if (over) return;
    d = abs(d);
    bool inside = (min(min(-w[1], -w[2]), -w[3]) >= -1e-6).item<int>();
    if (!inside)
        return;
//cout << d << endl;
    if (is_free(node)) {
        int side = (dot(n, node->n)>=0).item<int>() ? 0 : 1;
        ::node_prox[side][get_index(node, *::meshes)].add(d, (Face*)face);
    }
    if (is_free(face)) {
        int side = (dot(-n, face->n)>=0).item<int>() ? 0 : 1;
        ::face_prox[side][get_index(face, *::meshes)].add(d, (Node*)node);
    }
}

void obs_add_proximity (const Node *node, const Face *face, double thick) {
    if (node == face->v[0]->node
     || node == face->v[1]->node
     || node == face->v[2]->node)
        return;
    //if (!overlap(node_box(node, true), face_box(face, true), 2*::magic.repulsion_thickness.item<double>()))
    //    return;
    // cout << "obs_add_proximity ------- face " << endl;
    Tensor n;
    Tensor w[4];
    w[0]=w[1]=w[2]=w[3]=ZERO;
    n=ZERO3;
    bool over = false;
    Tensor d = signed_vf_distance(node->x, face->v[0]->node->x,
                                  face->v[1]->node->x, face->v[2]->node->x,
                                  &n, w, thick, over);
    if (over) return;
    d = abs(d);
    bool inside = (min(min(-w[1], -w[2]), -w[3]) >= -1e-6).item<int>();
    if (!inside)
        return;
//cout << d << endl;
    if (!is_free(node)) {

        // cout << find_mesh(face, *::obs_meshes) << endl;
        int side = (dot(n, node->n)>=0).item<int>() ? 0 : 1;
        ::obs_node_prox[side][get_index(node, *::obs_meshes)].add(d, (Face*)face);
    }
    if (!is_free(face)) {

        // cout << find_mesh(node, *::obs_meshes) << endl;
        int side = (dot(-n, face->n)>=0).item<int>() ? 0 : 1;
        ::obs_face_prox[side][get_index(face, *::obs_meshes)].add(d, (Node*)node);
    }
}

bool in_wedge (Tensor w, const Edge *edge0, const Edge *edge1) {
    Tensor x = (1-w)*edge0->n[0]->x + w*edge0->n[1]->x;
    bool in = true;
    for (int s = 0; s < 2; s++) {
        const Face *face = edge1->adjf[s];
        if (!face)
            continue;
        const Node *node0 = edge1->n[s], *node1 = edge1->n[1-s];
        Tensor e = node1->x - node0->x, n = face->n, r = x - node0->x;
        in &= (stp(e, n, r) >= 0).item<int>();
    }
    return in;
}

void obs_add_proximity (const Edge *edge0, const Edge *edge1, double thick) {
    if (edge0->n[0] == edge1->n[0] || edge0->n[0] == edge1->n[1]
     || edge0->n[1] == edge1->n[0] || edge0->n[1] == edge1->n[1])
        return;
    //if (!overlap(edge_box(edge0, true), edge_box(edge1, true), ::magic.repulsion_thickness.item<double>()))
    //    return;

    // cout << "obs_add_proximity ------- edge " << endl;

    Tensor n;
    Tensor w[4];
    w[0]=w[1]=w[2]=w[3]=ZERO;
    n=ZERO3;
    bool over = false;
    Tensor d = signed_ee_distance(edge0->n[0]->x, edge0->n[1]->x,
                                  edge1->n[0]->x, edge1->n[1]->x,
                                  &n, w, thick, over);
    if (over) return;
    d = abs(d);
    bool inside = ((min(min(w[0], w[1]), min(-w[2], -w[3])) >= -1e-6).item<int>()
                   && in_wedge(w[1], edge0, edge1)
                   && in_wedge(-w[3], edge1, edge0));
    if (!inside)
        return;
//cout << "good" << endl;
    if (!is_free(edge0)) {

        // cout << find_mesh(edge1, *::obs_meshes) << endl;
        Tensor edge0n = edge0->n[0]->n + edge0->n[1]->n;
        int side = (dot(n, edge0n)>=0).item<int>() ? 0 : 1;
        ::obs_edge_prox[side][get_index(edge0, *::obs_meshes)].add(d, (Edge*)edge1);
    }
    if (!is_free(edge1)) {

        // cout << find_mesh(edge0, *::obs_meshes) << endl;
        Tensor edge1n = edge1->n[0]->n + edge1->n[1]->n;
        int side = (dot(-n, edge1n)>=0).item<int>() ? 0 : 1;
        ::obs_edge_prox[side][get_index(edge1, *::obs_meshes)].add(d, (Edge*)edge0);
    }
}

void add_proximity (const Edge *edge0, const Edge *edge1, double thick) {
    if (edge0->n[0] == edge1->n[0] || edge0->n[0] == edge1->n[1]
     || edge0->n[1] == edge1->n[0] || edge0->n[1] == edge1->n[1])
        return;
    //if (!overlap(edge_box(edge0, true), edge_box(edge1, true), ::magic.repulsion_thickness.item<double>()))
    //    return;


    Tensor n;
    Tensor w[4];
    w[0]=w[1]=w[2]=w[3]=ZERO;
    n=ZERO3;
    bool over = false;
    Tensor d = signed_ee_distance(edge0->n[0]->x, edge0->n[1]->x,
                                  edge1->n[0]->x, edge1->n[1]->x,
                                  &n, w, thick, over);
    if (over) return;
    d = abs(d);
    bool inside = ((min(min(w[0], w[1]), min(-w[2], -w[3])) >= -1e-6).item<int>()
                   && in_wedge(w[1], edge0, edge1)
                   && in_wedge(-w[3], edge1, edge0));
    if (!inside)
        return;
//cout << "good" << endl;
    if (is_free(edge0)) {
        Tensor edge0n = edge0->n[0]->n + edge0->n[1]->n;
        int side = (dot(n, edge0n)>=0).item<int>() ? 0 : 1;
        ::edge_prox[side][get_index(edge0, *::meshes)].add(d, (Edge*)edge1);
    }
    if (is_free(edge1)) {
        Tensor edge1n = edge1->n[0]->n + edge1->n[1]->n;
        int side = (dot(-n, edge1n)>=0).item<int>() ? 0 : 1;
        ::edge_prox[side][get_index(edge1, *::meshes)].add(d, (Edge*)edge0);
    }
}

Tensor area (const Node *node);
Tensor area (const Edge *edge);
Tensor area (const Face *face);

Constraint *make_constraint (const Node *node, const Face *face,
                             Tensor mu, Tensor mu_obs) {
    // cout << "make_constraint face " << endl;
    IneqCon *con = new IneqCon;
    con->nodes[0] = (Node*)node;
    con->nodes[1] = (Node*)face->v[0]->node;
    con->nodes[2] = (Node*)face->v[1]->node;
    con->nodes[3] = (Node*)face->v[2]->node;
    for (int n = 0; n < 4; n++)
        con->free[n] = is_free(con->nodes[n]);
    Tensor a = min(area(node), area(face));
    con->stiff = ::magic.collision_stiffness*a;
    bool over;
    Tensor d = signed_vf_distance(con->nodes[0]->x, con->nodes[1]->x,
                                  con->nodes[2]->x, con->nodes[3]->x,
                                  &con->n, con->w, 100, over);
    if ((d < 0).item<int>())
        con->n = -con->n;
    con->mu = (!is_free(node) || !is_free(face)) ? mu_obs : mu;
    return con;
}

Constraint *make_constraint (const Edge *edge0, const Edge *edge1,
                             Tensor mu, Tensor mu_obs) {
    // cout << "make_constraint edge " << endl; 
    IneqCon *con = new IneqCon;
    con->nodes[0] = (Node*)edge0->n[0];
    con->nodes[1] = (Node*)edge0->n[1];
    con->nodes[2] = (Node*)edge1->n[0];
    con->nodes[3] = (Node*)edge1->n[1];
    for (int n = 0; n < 4; n++)
        con->free[n] = is_free(con->nodes[n]);
    Tensor a = min(area(edge0), area(edge1));
    con->stiff = ::magic.collision_stiffness*a;
    bool over;
    Tensor d = signed_ee_distance(con->nodes[0]->x, con->nodes[1]->x,
                                  con->nodes[2]->x, con->nodes[3]->x,
                                  &con->n, con->w, 100, over);
    if ((d < 0).item<int>())
        con->n = -con->n;
    con->mu = (!is_free(edge0) || !is_free(edge1)) ? mu_obs : mu;
    return con;
}

Tensor area (const Node *node) {
    if (is_free(node))
        return node->a;
    Tensor a = ZERO;
    for (int v = 0; v < node->verts.size(); v++)
        for (int f = 0; f < node->verts[v]->adjf.size(); f++)
            a = a + area(node->verts[v]->adjf[f])/3;
    return a;
}

Tensor area (const Edge *edge) {
    Tensor a = ZERO;
    if (edge->adjf[0])
        a = a + area(edge->adjf[0])/3;
    if (edge->adjf[1])
        a = a + area(edge->adjf[1])/3;
    return a;
}

Tensor area (const Face *face) {
    if (is_free(face))
        return face->a;
    const Tensor &x0 = face->v[0]->node->x, &x1 = face->v[1]->node->x,
               &x2 = face->v[2]->node->x;
    return norm(cross(x1-x0, x2-x0))/2;
}
