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

#ifndef PHYSICS_HPP
#define PHYSICS_HPP

#include "cloth.hpp"
#include "obstacle.hpp"
#include "geometry.hpp"
#include "simulation.hpp"
#include "taucs.hpp"
#include <vector>
using torch::Tensor;

template <Space s>
Tensor internal_energy (const Cloth &cloth);

Tensor constraint_energy (const std::vector<Constraint*> &cons);

Tensor external_energy (const Cloth &cloth, const Tensor &gravity,
                        const Wind &wind);

// A += dt^2 dF/dx; b += dt F + dt^2 dF/dx v
// also adds damping terms
// if dt == 0, just does A += dF/dx; b += F instead, no damping
template <Space s>
void add_internal_forces (const Cloth &cloth, SpMat &A,
                          Tensor &b, Tensor dt);

void add_constraint_forces (const Cloth &cloth,
                            const std::vector<Constraint*> &cons,
                            SpMat &A, Tensor &b, Tensor dt);

void add_external_forces (const Cloth &cloth, const Tensor &gravity,
                          const Wind &wind, Tensor &fext,
                          Tensor &Jext);

void obs_add_external_forces (const Obstacle &obstacle, const Tensor &gravity,
                          const Wind &wind, Tensor &fext,
                          Tensor &Jext);

void add_morph_forces (const Cloth &cloth, const Morph &morph, Tensor t,
                       Tensor dt,
                       Tensor &fext, Tensor &Jext);

void implicit_update (Cloth &cloth, const Tensor &fext,
                      const Tensor &Jext,
                      const std::vector<Constraint*> &cons, Tensor dt,
                      bool update_positions=true);

void obs_implicit_update (Obstacle &obstacle, const vector<Mesh*> &obs_meshes, const Tensor &fext,
                      const Tensor &Jext,
                      const vector<Constraint*> &cons, Tensor dt,
                      bool update_positions);

#endif
