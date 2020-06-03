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

#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP

#include "mesh.hpp"
#include "spline.hpp"
#include "util.hpp"
#include "vectors.hpp"
#include <map>
#include <vector>
using torch::Tensor;

typedef std::map<Node*,Tensor> MeshGrad;
typedef std::map<std::pair<Node*,Node*>,Tensor> MeshHess;

struct Constraint {
    virtual ~Constraint () {};
    virtual Tensor value (int *sign=NULL) = 0;
    virtual MeshGrad gradient () = 0;
    virtual MeshGrad project () = 0;
    // energy function
    virtual Tensor energy (Tensor value) = 0;
    virtual Tensor energy_grad (Tensor value) = 0;
    virtual Tensor energy_hess (Tensor value) = 0;
    // frictional force
    virtual MeshGrad friction (Tensor dt, MeshHess &jac) = 0;
};

struct EqCon: public Constraint {
    // n . (node->x - x) = 0
    Node *node;
    Tensor x, n;
    Tensor stiff;
    Tensor value (int *sign=NULL);
    MeshGrad gradient ();
    MeshGrad project ();
    Tensor energy (Tensor value);
    Tensor energy_grad (Tensor value);
    Tensor energy_hess (Tensor value);
    MeshGrad friction (Tensor dt, MeshHess &jac);
};

struct GlueCon: public Constraint {
    Node *nodes[2];
    Tensor n;
    Tensor stiff;
    Tensor value (int *sign=NULL);
    MeshGrad gradient ();
    MeshGrad project ();
    Tensor energy (Tensor value);
    Tensor energy_grad (Tensor value);
    Tensor energy_hess (Tensor value);
    MeshGrad friction (Tensor dt, MeshHess &jac);
};

struct IneqCon: public Constraint {
    // n . sum(w[i] verts[i]->x) >= 0
    Node *nodes[4];
    Tensor w[4];
    bool free[4];
    Tensor n;
    Tensor a; // area
    Tensor mu; // friction
    Tensor stiff;
    Tensor value (int *sign=NULL);
    MeshGrad gradient ();
    MeshGrad project ();
    Tensor energy (Tensor value);
    Tensor energy_grad (Tensor value);
    Tensor energy_hess (Tensor value);
    MeshGrad friction (Tensor dt, MeshHess &jac);
};

#endif
