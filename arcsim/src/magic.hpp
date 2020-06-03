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

#ifndef MAGIC_HPP
#define MAGIC_HPP

#include "vectors.hpp"
using torch::Tensor;

// Magic numbers and other hacks

struct Magic {
    bool fixed_high_res_mesh;
    Tensor handle_stiffness, collision_stiffness;
    Tensor repulsion_thickness, projection_thickness;
    Tensor edge_flip_threshold;
    Tensor rib_stiffening;
    Tensor rigid_damping;
    bool combine_tensors;
    bool preserve_creases;
    Magic ():
        fixed_high_res_mesh(false),
        handle_stiffness(ONE*1e3),
        collision_stiffness(ONE*1e9),
        repulsion_thickness(ONE*1e-3),
        projection_thickness(ONE*1e-4),
        edge_flip_threshold(ONE*1e-2),
        rigid_damping(ONE*0.99),
        rib_stiffening(ONE),
        combine_tensors(true),
        preserve_creases(false) {}
};

extern Magic magic;

#endif
