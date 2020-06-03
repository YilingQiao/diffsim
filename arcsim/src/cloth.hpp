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

#ifndef CLOTH_HPP
#define CLOTH_HPP

#include "dde.hpp"
#include "mesh.hpp"
using torch::Tensor;

struct Cloth {
    Mesh mesh;
    struct Material {
        Tensor density, densityori; // area density
        StretchingSamples stretching;
        StretchingData stretchingori;
        BendingData bending, bendingori;
        Tensor damping; // stiffness-proportional damping coefficient
        Tensor strain_min, strain_max; // strain limits
        Tensor yield_curv, weakening; // plasticity parameters
    };
    std::vector<Material*> materials;
    struct Remeshing {
        Tensor refine_angle, refine_compression, refine_velocity;
        Tensor size_min, size_max; // size limits
        Tensor aspect_min; // aspect ratio control
    } remeshing;
};

void compute_masses (Cloth &cloth);

#endif
