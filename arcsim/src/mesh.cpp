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

#include "mesh.hpp"
#include "geometry.hpp"
#include "util.hpp"
#include <assert.h>
#include <cstdlib>
using namespace std;
using torch::Tensor;

template <typename T1, typename T2> void check (const T1 *p1, const T2 *p2,
                                                const vector<T2*> &v2) {
    if (p2 && find((T2*)p2, v2) == -1) {
        cout << p1 << "'s adjacent " << p2 << " is not accounted for" << endl;
        abort();
    }
}
template <typename T1, typename T2> void not_null (const T1 *p1, const T2 *p2) {
    if (!p2) {
        cout << "adjacent to " << p1 << " is null " << p2 << endl;
        abort();
    }
}
template <typename T1, typename T2> void not_any_null
    (const T1 *p1, T2 *const*p2, int n) {
    bool any_null = false;
    for (int i = 0; i < n; i++) if (!p2[i]) any_null = true;
    if (any_null) {
        cout << "adjacent to " << p1 << " one of these is null" << endl;
        for (int i = 0; i < n; i++) cout << p2[i] << endl;
        abort();
    }
}
template <typename T1, typename T2> void not_all_null
    (const T1 *p1, T2 *const*p2, int n) {
    bool all_null = true;
    for (int i = 0; i < n; i++) if (p2[i]) all_null = false;
    if (all_null) {
        cout << "adjacent to " << p1 << " all of these are null" << endl;
        for (int i = 0; i < n; i++) cout << p2[i] << endl;
        abort();
    }
}

bool check_that_pointers_are_sane (const Mesh &mesh) {
    for (int v = 0; v < mesh.verts.size(); v++) {
        const Vert *vert = mesh.verts[v];
        not_null(vert, vert->node);
        check(vert, vert->node, mesh.nodes);
        if (find((Vert*)vert, vert->node->verts) == -1) {
            cout << "vert " << vert << "'s node " << vert->node
                 << " doesn't contain it" << endl;
            abort();
        }
        for (int i = 0; i < vert->adjf.size(); i++)
            check(vert, vert->adjf[i], mesh.faces);
    }
    for (int n = 0; n < mesh.nodes.size(); n++) {
        const Node *node = mesh.nodes[n];
        for (int i = 0; i < node->verts.size(); i++)
            check(node, node->verts[i], mesh.verts);
        for (int i = 0; i < 2; i++)
            check(node, node->adje[i], mesh.edges);
    }
    for (int e = 0; e < mesh.edges.size(); e++) {
        const Edge *edge = mesh.edges[e];
        for (int i = 0; i < 2; i++)
            check(edge, edge->n[i], mesh.nodes);
        not_any_null(edge, edge->n, 2);
        for (int i = 0; i < 2; i++)
            check(edge, edge->adjf[i], mesh.faces);
        not_all_null(edge, edge->adjf, 2);
    }
    for (int f = 0; f < mesh.faces.size(); f++) {
        const Face *face = mesh.faces[f];
        for (int i = 0; i < 3; i++)
            check(face, face->v[i], mesh.verts);
        not_any_null(face, face->v, 3);
        for (int i = 0; i < 3; i++)
            check(face, face->adje[i], mesh.edges);
        not_any_null(face, face->adje, 3);
    }
    return true;
}

bool check_that_contents_are_sane (const Mesh &mesh) {
    // // TODO
    // for (int v = 0; v < mesh.verts.size(); v++) {
    //     const Vert *vert = mesh.verts[v];
    //     if (!isfinite(norm2(vert->x + vert->v + vert->n) + vert->a)) {
    //         cout << "Vertex " << name(vert) << " is " << vert->x << " "
    //              << vert->v << " " << vert->n << " " << vert->a << endl;
    //         return false;
    //     }
    // }
    // for (int f = 0; f < mesh.faces.size(); f++) {
    //     const Face *face = mesh.faces[f];
    //     if (!isfinite(norm2(face->n) + face->a)) {
    //         cout << "Face " << name(face) << " is " << face->n << " "
    //              << face->a << endl;
    //         return false;
    //     }
    // }
    return true;
}

// Material space data

void compute_ms_data (Face* face) {
    face->Dm = torch::stack({face->v[1]->u - face->v[0]->u,
                      face->v[2]->u - face->v[0]->u},1);
    face->a = det(face->Dm)/2;
    if ((face->a == ZERO).item<int>())
        face->invDm = torch::zeros({2,2},TNOPT);
    else
        face->invDm = face->Dm.inverse();
}

void compute_ms_data (Edge* edge) {
    edge->l = ZERO;
    for (int s = 0; s < 2; s++)
        if (edge->adjf[s])
            edge->l = edge->l + norm(edge_vert(edge,s,0)->u - edge_vert(edge,s,1)->u);
    if (edge->adjf[0] && edge->adjf[1])
        edge->l = edge->l / 2;

    if (!edge->adjf[0] || !edge->adjf[1])
        return;
    edge->ldaa = edge->l/(edge->adjf[0]->a + edge->adjf[1]->a);
    Tensor du0 = edge_vert(edge, 0, 1)->u - edge_vert(edge, 0, 0)->u;
    Tensor du1 = edge_vert(edge, 1, 1)->u - edge_vert(edge, 1, 0)->u;
    edge->bias_angle = torch::stack({atan2(du0[1], du0[0]), atan2(du1[1], du1[0])})*(4/M_PI)-1;
}

void compute_ms_data (Vert* vert) {
    vert->a = ZERO;
    const vector<Face*> &adjfs = vert->adjf;
    for (int i = 0; i < adjfs.size(); i++) {
        Face const* face = adjfs[i];
        vert->a = vert->a + face->a/3;
    }
}

void compute_ms_data (Node* node) {
    node->a = ZERO;
    for (int v = 0; v < node->verts.size(); v++)
        node->a = node->a + node->verts[v]->a;
}

void compute_ms_data (Mesh &mesh) {
    for (int f = 0; f < mesh.faces.size(); f++)
        compute_ms_data(mesh.faces[f]);
    for (int e = 0; e < mesh.edges.size(); e++)
        compute_ms_data(mesh.edges[e]);
    for (int v = 0; v < mesh.verts.size(); v++)
        compute_ms_data(mesh.verts[v]);
    for (int n = 0; n < mesh.nodes.size(); n++)
        compute_ms_data(mesh.nodes[n]);
    // now we need to recompute world-space data
    compute_ws_data(mesh);
}

// World-space data

void compute_ws_data (Face* face) {
    const Tensor &x0 = face->v[0]->node->x,
               &x1 = face->v[1]->node->x,
               &x2 = face->v[2]->node->x;
    face->n = normalize(cross(x1-x0, x2-x0));
    // Mat3x2 F = derivative(x0, x1, x2, face);
    // SVD<3,2> svd = singular_value_decomposition(F);
    // Mat3x2 Vt_ = 0;
    // for (int i = 0; i < 2; i++)
    //     for (int j = 0; j < 2; j++)
    //         Vt_(i,j) = svd.Vt(i,j);
    // face->R = svd.U*Vt_;
    // face->F = svd.Vt.t()*diag(svd.s)*svd.Vt;
}

void compute_ws_data (Edge *edge) {
    edge->theta = dihedral_angle<WS>(edge);
}

void compute_ws_data (Node* node) {
    node->n = ZERO3;
    for (int v = 0; v < node->verts.size(); v++) {
        const Vert *vert = node->verts[v];
        const vector<Face*> &adjfs = vert->adjf;
        for (int i = 0; i < adjfs.size(); i++) {
            Face const* face = adjfs[i];
            int j = find(vert, face->v), j1 = (j+1)%3, j2 = (j+2)%3;
            Tensor e1 = face->v[j1]->node->x - node->x,
                   e2 = face->v[j2]->node->x - node->x;
            node->n = node->n + cross(e1,e2)/(2*norm2(e1)*norm2(e2));//face->n;//
        }
    }
    node->n = normalize(node->n);
}

void compute_ws_data (Mesh &mesh) {
    for (int f = 0; f < mesh.faces.size(); f++)
        compute_ws_data(mesh.faces[f]);
    if (mesh.isCloth)
        for (int e = 0; e < mesh.edges.size(); e++)
            compute_ws_data(mesh.edges[e]);
    for (int n = 0; n < mesh.nodes.size(); n++)
        compute_ws_data(mesh.nodes[n]);
}

// Mesh operations

template <> const vector<Vert*> &get (const Mesh &mesh) {return mesh.verts;}
template <> const vector<Node*> &get (const Mesh &mesh) {return mesh.nodes;}
template <> const vector<Edge*> &get (const Mesh &mesh) {return mesh.edges;}
template <> const vector<Face*> &get (const Mesh &mesh) {return mesh.faces;}

Edge *get_edge (const Node *n0, const Node *n1) {
    for (int e = 0; e < n0->adje.size(); e++) {
        Edge *edge = n0->adje[e];
        if (edge->n[0] == n1 || edge->n[1] == n1)
            return edge;
    }
    return NULL;
}

Vert *edge_vert (const Edge *edge, int side, int i) {
    Face *face = (Face*)edge->adjf[side];
    if (!face)
        return NULL;
    for (int j = 0; j < 3; j++)
        if (face->v[j]->node == edge->n[i])
            return face->v[j];
    return NULL;
}

Vert *edge_opp_vert (const Edge *edge, int side) {
    Face *face = (Face*)edge->adjf[side];
    if (!face)
        return NULL;
    for (int j = 0; j < 3; j++)
        if (face->v[j]->node == edge->n[side])
            return face->v[PREV(j)];
    return NULL;
}

void connect (Vert *vert, Node *node) {
    vert->node = node;
    include(vert, node->verts);
}

void Mesh::add (Vert *vert) {
    verts.push_back(vert);
    vert->node = NULL;
    vert->adjf.clear();
    vert->index = verts.size()-1;
}

void Mesh::remove (Vert* vert) {
    if (!vert->adjf.empty()) {
        cout << "Error: can't delete vert " << vert << " as it still has "
             << vert->adjf.size() << " faces attached to it." << endl;
        return;
    }
    exclude(vert, verts);
}

void Mesh::add (Node *node) {
    nodes.push_back(node);
    node->preserve = false;
    node->index = nodes.size()-1;
    node->adje.clear();
    for (int v = 0; v < node->verts.size(); v++)
        node->verts[v]->node = node;
}

void Mesh::remove (Node* node) {
    if (!node->adje.empty()) {
        cout << "Error: can't delete node " << node << " as it still has "
             << node->adje.size() << " edges attached to it." << endl;
        return;
    }
    exclude(node, nodes);
}

void Mesh::add (Edge *edge) {
    edges.push_back(edge);
    edge->adjf[0] = edge->adjf[1] = NULL;
    edge->index = edges.size()-1;
    include(edge, edge->n[0]->adje);
    include(edge, edge->n[1]->adje);
}

void Mesh::remove (Edge *edge) {
    if (edge->adjf[0] || edge->adjf[1]) {
        cout << "Error: can't delete edge " << edge
             << " as it still has a face attached to it." << endl;
        return;
    }
    exclude(edge, edges);
    exclude(edge, edge->n[0]->adje);
    exclude(edge, edge->n[1]->adje);
}

void add_edges_if_needed (Mesh &mesh, const Face *face) {
    for (int i = 0; i < 3; i++) {
        Node *n0 = face->v[i]->node, *n1 = face->v[NEXT(i)]->node;
        if (get_edge(n0, n1) == NULL)
            mesh.add(new Edge(n0, n1));
    }
}

void Mesh::add (Face *face) {
    faces.push_back(face);
    face->index = faces.size()-1;
    // adjacency
    add_edges_if_needed(*this, face);
    for (int i = 0; i < 3; i++) {
        Vert *v0 = face->v[NEXT(i)], *v1 = face->v[PREV(i)];
        include(face, v0->adjf);
        Edge *e = get_edge(v0->node, v1->node);
        face->adje[i] = e;
        int side = e->n[0]==v0->node ? 0 : 1;
        e->adjf[side] = face;
    }
}

void Mesh::remove (Face* face) {
    exclude(face, faces);
    // adjacency
    for (int i = 0; i < 3; i++) {
        Vert *v0 = face->v[NEXT(i)];
        exclude(face, v0->adjf);
        Edge *e = face->adje[i];
        int side = e->n[0]==v0->node ? 0 : 1;
        e->adjf[side] = NULL;
    }
}

void update_indices (Mesh &mesh) {
    for (int v = 0; v < mesh.verts.size(); v++)
        mesh.verts[v]->index = v;
    for (int f = 0; f < mesh.faces.size(); f++)
        mesh.faces[f]->index = f;
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->index = n;
    for (int e = 0; e < mesh.edges.size(); e++)
        mesh.edges[e]->index = e;
}

void mark_nodes_to_preserve (Mesh &mesh) {
    for (int n = 0; n < mesh.nodes.size(); n++) {
        Node *node = mesh.nodes[n];
        if (is_seam_or_boundary(node) || node->label)
            node->preserve = true;
    }
    for (int e = 0; e < mesh.edges.size(); e++) {
        Edge *edge = mesh.edges[e];
        if (edge->label) {
            edge->n[0]->preserve = true;
            edge->n[1]->preserve = true;
        }
    }
}

void apply_transformation_onto (const Mesh &start_state, Mesh &onto,
                                const Transformation &tr) {
    for (int n = 0; n < onto.nodes.size(); n++)
        onto.nodes[n]->x = tr.apply(start_state.nodes[n]->x);
    compute_ws_data(onto);
}

void opt_apply_transformation_onto (const Mesh &start_state, Mesh &onto,
                                const Transformation &tr) {
    for (int n = 0; n < onto.nodes.size(); n++)
        onto.nodes[n]->x = tr.apply(start_state.nodes[n]->x);
}

void apply_transformation (Mesh& mesh, const Transformation& tr) {
    apply_transformation_onto(mesh, mesh, tr);
}

void update_x0 (Mesh &mesh) {
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->x0 = mesh.nodes[n]->x;
}

Mesh deep_copy (const Mesh &mesh0) {
    Mesh mesh1;
    mesh1.isCloth = mesh0.isCloth;
    if (mesh0.dummy_node) {
        const Node *node0 = mesh0.dummy_node;
        mesh1.dummy_node = new Node(node0->x, node0->v, node0->label);
    }

    for (int v = 0; v < mesh0.verts.size(); v++) {
        const Vert *vert0 = mesh0.verts[v];
        Vert *vert1 = new Vert(vert0->u, vert0->label);
        mesh1.add(vert1);
    }
    for (int n = 0; n < mesh0.nodes.size(); n++) {
        const Node *node0 = mesh0.nodes[n];
        Node *node1 = new Node(node0->x, node0->v, node0->label);
        node1->preserve = node0->preserve;
        node1->verts.resize(node0->verts.size());
        for (int v = 0; v < node0->verts.size(); v++)
            node1->verts[v] = mesh1.verts[node0->verts[v]->index];
        mesh1.add(node1);
    }
    for (int e = 0; e < mesh0.edges.size(); e++) {
        const Edge *edge0 = mesh0.edges[e];
        Edge *edge1 = new Edge(mesh1.nodes[edge0->n[0]->index],
                               mesh1.nodes[edge0->n[1]->index],
                               edge0->label);
        mesh1.add(edge1);
    }
    for (int f = 0; f < mesh0.faces.size(); f++) {
        const Face *face0 = mesh0.faces[f];
        Face *face1 = new Face(mesh1.verts[face0->v[0]->index],
                               mesh1.verts[face0->v[1]->index],
                               mesh1.verts[face0->v[2]->index],
                               face0->label);
        mesh1.add(face1);
    }
    compute_ms_data(mesh1);
    return mesh1;
}

void delete_mesh (Mesh &mesh) {
    for (int v = 0; v < mesh.verts.size(); v++)
        delete mesh.verts[v];
    for (int n = 0; n < mesh.nodes.size(); n++)
        delete mesh.nodes[n];
    for (int e = 0; e < mesh.edges.size(); e++)
        delete mesh.edges[e];
    for (int f = 0; f < mesh.faces.size(); f++)
        delete mesh.faces[f];
    if (mesh.dummy_node)
        delete mesh.dummy_node;
    mesh.verts.clear();
    mesh.nodes.clear();
    mesh.edges.clear();
    mesh.faces.clear();
}
