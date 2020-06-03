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

#include "bah.hpp"
using torch::Tensor;

Box &Box::operator+= (const Tensor &u) {
    umin = min(umin, u);
    umax = max(umax, u);
    return *this;
}

Box &Box::operator+= (const Box &box) {
    umin = min(umin, box.umin);
    umax = max(umax, box.umax);
    return *this;
}

bool Box::overlaps (const Box &box) const {
    for (int i = 0; i < 2; i++) {
        if ((umin[i] > box.umax[i]).item<int>()) return false;
        if ((umax[i] < box.umin[i]).item<int>()) return false;
    }
    return true;
}

Tensor Box::size () const {
    return umax - umin;
}

Tensor Box::center () const {
    return (umin + umax)/2.;
}

Box vert_box (const Vert *vert) {
    return Box(vert->u);
}

Box face_box (const Face *face) {
    Box box;
    for (int v = 0; v < 3; v++)
        box += face->v[v]->u;
    return box;
}

struct Aap {
	int xy;
	Tensor p;
	Aap (const Box &total) {
		Tensor center = total.center();
        Tensor size = total.size();
		xy = ((size[0]>=size[1]).item<int>()) ? 0 : 1;
		p = center[xy];
	}
	bool inside (const Tensor &mid) const {
		return (mid[xy] > p).item<int>();
	}
};

BahNode *new_bah_tree (const Mesh &mesh) {
	Box total;
	int count;
    int num_vtx = mesh.verts.size(),
        num_tri = mesh.faces.size();
    for (unsigned int i=0; i<num_vtx; i++)
        total += mesh.verts[i]->u;
    count = num_tri;
	Box *tri_boxes = new Box[count];
	Tensor *tri_centers = new Tensor[count];
	Aap pln(total);
	Face **face_buffer = new Face*[count];
	unsigned int left_idx = 0, right_idx = count;
	unsigned int tri_idx = 0;
	for (unsigned int i=0; i<num_tri; i++) {
		Tensor &p1 = mesh.faces[i]->v[0]->u;
		Tensor &p2 = mesh.faces[i]->v[1]->u;
		Tensor &p3 = mesh.faces[i]->v[2]->u;
        tri_centers[i] = (p1 + p2 + p3)/3.;
		if (pln.inside(tri_centers[i]))
			face_buffer[left_idx++] = mesh.faces[i];
		else
			face_buffer[--right_idx] = mesh.faces[i];
		tri_boxes[i] += p1;
		tri_boxes[i] += p2;
		tri_boxes[i] += p3;
	}
    BahNode *root = new BahNode;
	root->box = total;
	if (count == 1) {
		root->face = mesh.faces[0];
		root->left = root->right = NULL;
	} else {
		if (left_idx == 0 || left_idx == count)
			left_idx = count/2;
		root->left = new BahNode(root, face_buffer, left_idx,
                                 tri_boxes, tri_centers);
		root->right = new BahNode(root, face_buffer+left_idx, count-left_idx,
                                  tri_boxes, tri_centers);
	}
	delete [] tri_boxes;
	delete [] tri_centers;
    delete [] face_buffer;
    return root;
}

BahNode::BahNode ():
    face(NULL), left(NULL), right(NULL), parent(NULL) {
}

BahNode::~BahNode () {
    if (left) delete left;
    if (right) delete right;
}

BahNode::BahNode (BahNode *parent, Face *face, const Box &box):
    left(NULL), right(NULL), parent(parent), face(face), box(box) {
}

BahNode::BahNode (BahNode *parent, Face **lst, unsigned int lst_num,
                  Box *tri_boxes, Tensor *tri_centers) {
    BahNode *node = new BahNode;
	assert(lst_num > 0);
	left = right = NULL;
	parent = parent;
	face = NULL;
	if (lst_num == 1) {
		face = lst[0];
		box = tri_boxes[lst[0]->index];
	} else { // try to split them
		for (unsigned int t=0; t<lst_num; t++) {
			int i=lst[t]->index;
			box += tri_boxes[i];
		}
		if (lst_num == 2) { // must split it!
			left = new BahNode(this, lst[0], tri_boxes[lst[0]->index]);
			right = new BahNode(this, lst[1], tri_boxes[lst[1]->index]);
		} else {
			Aap pln(box);
			unsigned int left_idx = 0, right_idx = lst_num-1;
			for (unsigned int t=0; t<lst_num; t++) {
				int i=lst[left_idx]->index;
				if (pln.inside(tri_centers[i]))
					left_idx++;
				else {// swap it
					Face *tmp = lst[left_idx];
					lst[left_idx] = lst[right_idx];
					lst[right_idx--] = tmp;
				}
			}
			int hal = lst_num/2;
			if (left_idx == 0 || left_idx == lst_num) {
				left = new BahNode(this, lst, hal, tri_boxes, tri_centers);
				right = new BahNode(this, lst+hal, lst_num-hal,
                                    tri_boxes, tri_centers);
			} else {
				left = new BahNode(this, lst, left_idx, tri_boxes, tri_centers);
				right = new BahNode(this, lst+left_idx, lst_num-left_idx,
                                    tri_boxes, tri_centers);
			}
		}
	}
}

void delete_bah_tree (BahNode *root) {
    delete root;
}

void for_overlapping_faces (Face *face, const Box &box, const BahNode *node,
                            BahCallback callback);

void for_overlapping_faces (Face *face, const BahNode *node,
                            BahCallback callback) {
    for_overlapping_faces(face, face_box(face), node, callback);
}

void for_overlapping_faces (Face *face, const Box &box, const BahNode *node,
                            BahCallback callback) {
    if (!box.overlaps(node->box))
        return;
    if (node->face) {
        callback(face, node->face);
    } else {
        for_overlapping_faces(face, box, node->left, callback);
        for_overlapping_faces(face, box, node->right, callback);
    }
}
