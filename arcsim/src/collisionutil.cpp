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

#include "collisionutil.hpp"

#include "simulation.hpp"
#include <omp.h>
using namespace std;
using torch::Tensor;

void collect_leaves (BVHNode *node, vector<BVHNode*> &leaves);

AccelStruct::AccelStruct (const Mesh &mesh, bool ccd):
    tree((Mesh&)mesh, ccd), root(tree._root), leaves(mesh.faces.size()) {
    if (root)
        collect_leaves(root, leaves);
}

void collect_leaves (BVHNode *node, vector<BVHNode*> &leaves) {
    if (node->isLeaf()) {
        int f = node->getFace()->index;
        if (f >= leaves.size())
            leaves.resize(f+1);
        leaves[f] = node;
    } else {
        collect_leaves(node->getLeftChild(), leaves);
        collect_leaves(node->getRightChild(), leaves);
    }
}

void update_accel_struct (AccelStruct &acc) {
    if (acc.root)
        acc.tree.refit();
}

void mark_descendants (BVHNode *node, bool active);
void mark_ancestors (BVHNode *node, bool active);

void mark_all_inactive (AccelStruct &acc) {
    if (acc.root)
        mark_descendants(acc.root, false);
}

void mark_active (AccelStruct &acc, const Face *face) {
    if (acc.root)
        mark_ancestors(acc.leaves[face->index], true);
}

void mark_descendants (BVHNode *node, bool active) {
    node->_active = active;
    if (!node->isLeaf()) {
        mark_descendants(node->_left, active);
        mark_descendants(node->_right, active);
    }
}

void mark_ancestors (BVHNode *node, bool active) {
    node->_active = active;
    if (!node->isRoot())
        mark_ancestors(node->_parent, active);
}

void for_overlapping_faces (BVHNode *node, double thickness,
                            BVHCallback callback) {
	if (node->isLeaf() || !node->_active)
		return;
	for_overlapping_faces(node->getLeftChild(), thickness, callback);
	for_overlapping_faces(node->getRightChild(), thickness, callback);
	for_overlapping_faces(node->getLeftChild(), node->getRightChild(),
                          thickness, callback);
}

void for_overlapping_faces (BVHNode *node0, BVHNode *node1, double thickness,
                            BVHCallback callback) {
    if (!node0->_active && !node1->_active)
        return;
    if (!overlap(node0->_box, node1->_box, thickness))
        return;
    if (node0->isLeaf() && node1->isLeaf()) {
        Face *face0 = node0->getFace(),
             *face1 = node1->getFace();
        callback(face0, face1);
    } else if (node0->isLeaf()) {
        for_overlapping_faces(node0, node1->getLeftChild(), thickness,callback);
        for_overlapping_faces(node0, node1->getRightChild(),thickness,callback);
    } else {
        for_overlapping_faces(node0->getLeftChild(), node1, thickness,callback);
        for_overlapping_faces(node0->getRightChild(), node1,thickness,callback);
    }
}

vector<BVHNode*> collect_upper_nodes (const vector<AccelStruct*> &accs, int n);

void for_overlapping_faces (const vector<AccelStruct*> &accs,
                            const vector<AccelStruct*> &obs_accs,
                            Tensor thickness0, BVHCallback callback,
                            bool parallel) {
    omp_set_num_threads(1);
    double thickness = thickness0.item<double>();
    int nnodes = (int)ceil(sqrt(2*omp_get_max_threads()));
    vector<BVHNode*> nodes = collect_upper_nodes(accs, nnodes);
    int nthreads = omp_get_max_threads();
    omp_set_num_threads(parallel ? omp_get_max_threads() : 1);
#pragma omp parallel for
    for (int n = 0; n < nodes.size(); n++) {
        for_overlapping_faces(nodes[n], thickness, callback);
        for (int m = 0; m < n; m++)
            for_overlapping_faces(nodes[n], nodes[m], thickness, callback);
        for (int o = 0; o < obs_accs.size(); o++)
            if (obs_accs[o]->root)
                for_overlapping_faces(nodes[n], obs_accs[o]->root, thickness,
                                      callback);
    }

    // cout << "obs_accs.size(): " << obs_accs.size() << endl;
    nodes = collect_upper_nodes(obs_accs, nnodes);
#pragma omp parallel for
    for (int n = 0; n < nodes.size(); n++) {
        for_overlapping_faces(nodes[n], thickness, callback);
        for (int m = 0; m < n; m++)
            for_overlapping_faces(nodes[n], nodes[m], thickness, callback);
        //for (int o = 0; o < obs_accs.size(); o++)
        //    if (obs_accs[o]->root)
        //        for_overlapping_faces(nodes[n], obs_accs[o]->root, thickness,
        //                              callback);
    }

    omp_set_num_threads(nthreads);
}

vector<BVHNode*> collect_upper_nodes (const vector<AccelStruct*> &accs,
                                      int nnodes) {
    vector<BVHNode*> nodes;
    for (int a = 0; a < accs.size(); a++)
        if (accs[a]->root)
            nodes.push_back(accs[a]->root);
    while (nodes.size() < nnodes) {
        vector<BVHNode*> children;
        for (int n = 0; n < nodes.size(); n++)
            if (nodes[n]->isLeaf())
                children.push_back(nodes[n]);
            else {
                children.push_back(nodes[n]->_left);
                children.push_back(nodes[n]->_right);
            }
        if (children.size() == nodes.size())
            break;
        nodes = children;
    }
    return nodes;
}

vector<AccelStruct*> create_accel_structs (const vector<Mesh*> &meshes,
                                           bool ccd) {
    vector<AccelStruct*> accs(meshes.size());
    for (int m = 0; m < meshes.size(); m++)
        accs[m] = new AccelStruct(*meshes[m], ccd);
    return accs;
}

void destroy_accel_structs (vector<AccelStruct*> &accs) {
    for (int a = 0; a < accs.size(); a++)
        delete accs[a];
}

template <typename Prim>
int find_mesh (const Prim *p, const vector<Mesh*> &meshes) {
    for (int m = 0; m < meshes.size(); m++) {
        const vector<Prim*> &ps = get<Prim>(*meshes[m]);
        if (p->index < ps.size() && p == ps[p->index])
            return m;
    }
    return -1;
}
template int find_mesh (const Vert*, const vector<Mesh*>&);
template int find_mesh (const Node*, const vector<Mesh*>&);
template int find_mesh (const Edge*, const vector<Mesh*>&);
template int find_mesh (const Face*, const vector<Mesh*>&);

int find_mesh_dummy (const Node *p, const vector<Mesh*> &meshes) {
    for (int m = 0; m < meshes.size(); m++) {
        if (meshes[m]->dummy_node == p)
            return m;
    }
    return -1;
}

const vector<Mesh*> *meshes, *obs_meshes;
