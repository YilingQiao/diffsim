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

#ifndef REMESH_HPP
#define REMESH_HPP

#include "mesh.hpp"
using torch::Tensor;

// Pointers are owned by the RemeshOp.
// Use done() and/or inverse().done() to free.

struct RemeshOp {
    std::vector<Vert*> added_verts, removed_verts;
    std::vector<Node*> added_nodes, removed_nodes;
    std::vector<Edge*> added_edges, removed_edges;
    std::vector<Face*> added_faces, removed_faces;
    bool empty () {return added_faces.empty() && removed_faces.empty();}
    RemeshOp inverse () const;
    void apply (Mesh &mesh) const;
    void done () const; // frees removed data
};
std::ostream &operator<< (std::ostream &out, const RemeshOp &op);

RemeshOp compose (const RemeshOp &op1, const RemeshOp &op2);

// These do not change the mesh directly,
// they return a RemeshOp that you can apply() to the mesh

RemeshOp split_edge (Edge *edge);

RemeshOp collapse_edge (Edge *edge, int which); // which end to delete

RemeshOp flip_edge (Edge *edge);

#endif
