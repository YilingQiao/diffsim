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

#include "popfilter.hpp"

#include "magic.hpp"
#include "optimization.hpp"
#include "physics.hpp"
#include "taucs.hpp"
#include <utility>
using namespace std;
using torch::Tensor;

// "rubber band" stiffness to stop vertices moving too far from initial position
static double mu;

struct PopOpt: public NLOpt {
    // we want F(x) = m a0, but F(x) = -grad E(x)
    // so grad E(x) + m a0 = 0
    // let's minimize E(x) + m a0 . (x - x0)
    // add spring to x0: minimize E(x) + m a0 . (x - x0) + mu (x - x0)^2/2
    // gradient: -F(x) + m a0 + mu (x - x0)
    // hessian: -J(x) + mu
    Cloth &cloth;
    Mesh &mesh;
    const vector<Constraint*> &cons;
    Tensor x0, a0;
    mutable Tensor f;
    mutable SpMat J;
    PopOpt (Cloth &cloth, const vector<Constraint*> &cons):
        cloth(cloth), mesh(cloth.mesh), cons(cons) {
        int nn = mesh.nodes.size();
        nvar = nn*3;
        x0 = torch::zeros({nn,3},TNOPT);
        a0 = torch::zeros({nn,3},TNOPT);
        for (int n = 0; n < nn; n++) {
            const Node *node = mesh.nodes[n];
            x0[n] = node->x;
            a0[n] = node->acceleration;
        }
        f = torch::zeros({nn,3},TNOPT);
        J = SpMat(nn,nn);//torch::zeros({nn*3,nn*3},TNOPT);
    }
    virtual void initialize (Tensor &x) const;
    virtual void precompute (const Tensor &x) const;
    virtual Tensor objective (const Tensor &x) const;
    virtual void gradient (const Tensor &x, Tensor &g) const;
    virtual bool hessian (const Tensor &x, SpMat &H) const;
    virtual void finalize (const Tensor &x) const;
};

void subtract_rigid_acceleration (const Mesh &mesh);

void apply_pop_filter (Cloth &cloth, const vector<Constraint*> &cons,
                       double regularization) {
    ::mu = regularization;
    // subtract_rigid_acceleration(cloth.mesh);
    // trust_region_method(PopOpt(cloth, cons), true);
cout << "pop filter"<<endl;
    line_search_newtons_method(PopOpt(cloth, cons), OptOptions().max_iter(10));
    compute_ws_data(cloth.mesh);
cout << "done pop filter"<<endl;
}

void PopOpt::initialize (Tensor &x) const {
    for (int n = 0; n < mesh.nodes.size(); n++)
        set_subvec(x, n, ZERO3);
}

void PopOpt::precompute (const Tensor &x) const {
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->x = x0[n] + get_subvec(x, n);
    // J = torch::zeros({nvar, nvar},TNOPT);
    int nn = mesh.nodes.size();
    f = torch::zeros({nn,3}, TNOPT);
    J = SpMat(nn,nn);
    add_internal_forces<WS>(cloth, J, f, ZERO);
    // cout << "------------------------------------------------------------"<<endl;
    add_constraint_forces(cloth, cons, J, f, ZERO);
}

Tensor PopOpt::objective (const Tensor &x) const {
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->x = x0[n] + get_subvec(x, n);
    Tensor e = internal_energy<WS>(cloth);
    e = e + constraint_energy(cons);
    for (int n = 0; n < mesh.nodes.size(); n++) {
        const Node *node = mesh.nodes[n];
        e = e + node->m*dot(node->acceleration, node->x - x0[n]);
        e = e + ::mu*norm2(node->x - x0[n])/2.;
    }
    return e;
}

void PopOpt::gradient (const Tensor &x, Tensor &g) const {
    for (int n = 0; n < mesh.nodes.size(); n++) {
        const Node *node = mesh.nodes[n];
        set_subvec(g, n, -f[n] + node->m*a0[n]
                         + ::mu*(node->x - x0[n]));
    }
}

bool PopOpt::hessian (const Tensor &x, SpMat &H) const {
    // H = J + torch::eye(nvar,TNOPT)*::mu;
    for (int i = 0; i < mesh.nodes.size(); i++) {
        const SpVec &Ji = J.rows[i];
        for (int jj = 0; jj < Ji.indices.size(); jj++) {
            int j = Ji.indices[jj];
            const Tensor &Jij = Ji.entries[jj];
            H(i,j) = Jij;
        }
        add_submat(H, i, i, EYE3*::mu);
    }
    return true;
}

void PopOpt::finalize (const Tensor &x) const {
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->x = x0[n] + get_subvec(x, n);
}
