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

#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include "mesh.hpp"
#include "util.hpp"
using torch::Tensor;

Tensor signed_vf_distance (const Tensor &x,
                           const Tensor &y0, const Tensor &y1, const Tensor &y2,
                           Tensor *n, Tensor *w, double thres, bool &over);
Tensor sub_signed_vf_distance (const Tensor &y0, const Tensor &y1, const Tensor &y2,
                           Tensor *n, Tensor *w, double thres, bool &over);

Tensor signed_ee_distance (const Tensor &x0, const Tensor &x1,
                           const Tensor &y0, const Tensor &y1,
                           Tensor *n, Tensor *w, double thres, bool &over);
Tensor sub_signed_ee_distance (const Tensor &x1mx0, const Tensor &y0mx0, const Tensor &y1mx0,
                           const Tensor &y0mx1, const Tensor &y1mx1, const Tensor &y1my0,
                           Tensor *n, Tensor *w, double thres, bool &over);

Tensor unsigned_vf_distance (const Tensor &x,
                             const Tensor &y0, const Tensor &y1, const Tensor &y2,
                             Tensor *n, Tensor *w);

Tensor unsigned_ee_distance (const Tensor &x0, const Tensor &x1,
                             const Tensor &y0, const Tensor &y1,
                             Tensor *n, Tensor *w);

Tensor get_barycentric_coords (const Tensor &point, const Face *face);

Face* get_enclosing_face (const Mesh& mesh, const Tensor& u,
                          Face *starting_face_hint = NULL);

enum Space {PS, WS}; // plastic space, world space

template <Space s> const Tensor &pos (const Node *node);
template <Space s> Tensor &pos (Node *node);
template <Space s> Tensor nor (const Face *face);
template <Space s> Tensor dihedral_angle (const Edge *edge);
template <Space s> Tensor curvature (const Face *face);

Tensor unwrap_angle (Tensor theta, Tensor theta_ref);

#endif
