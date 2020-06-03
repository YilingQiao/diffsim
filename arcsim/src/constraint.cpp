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

#include "constraint.hpp"

#include "magic.hpp"
using namespace std;
using torch::Tensor;

Tensor EqCon::value (int *sign) {
    if (sign) *sign = 0;
    return dot(n, node->x - x);
}
MeshGrad EqCon::gradient () {MeshGrad grad; grad[node] = n; return grad;}
MeshGrad EqCon::project () {return MeshGrad();}
Tensor EqCon::energy (Tensor value) {return stiff*sq(value)/2.;}
Tensor EqCon::energy_grad (Tensor value) {return stiff*value;}
Tensor EqCon::energy_hess (Tensor value) {return stiff;}
MeshGrad EqCon::friction (Tensor dt, MeshHess &jac) {return MeshGrad();}

Tensor GlueCon::value (int *sign) {
    if (sign) *sign = 0;
    return dot(n, nodes[1]->x - nodes[0]->x);
}
MeshGrad GlueCon::gradient () {
    MeshGrad grad;
    grad[nodes[0]] = -n;
    grad[nodes[1]] = n;
    return grad;
}
MeshGrad GlueCon::project () {return MeshGrad();}
Tensor GlueCon::energy (Tensor value) {return stiff*sq(value)/2.;}
Tensor GlueCon::energy_grad (Tensor value) {return stiff*value;}
Tensor GlueCon::energy_hess (Tensor value) {return stiff;}
MeshGrad GlueCon::friction (Tensor dt, MeshHess &jac) {return MeshGrad();}

Tensor IneqCon::value (int *sign) {
    if (sign)
        *sign = 1;
    Tensor d = ZERO;
    for (int i = 0; i < 4; i++)
        d = d + w[i]*dot(n, nodes[i]->x);
    d = d - ::magic.repulsion_thickness;
    return d;
}

MeshGrad IneqCon::gradient () {
    MeshGrad grad;
    for (int i = 0; i < 4; i++)
        grad[nodes[i]] = w[i]*n;
    return grad;
}

MeshGrad IneqCon::project () {
    Tensor d = value() + ::magic.repulsion_thickness - ::magic.projection_thickness;
    if ((d >= 0).item<int>())
        return MeshGrad();
    Tensor inv_mass = ZERO;
    for (int i = 0; i < 4; i++)
        if (free[i])
            inv_mass = inv_mass + sq(w[i])/nodes[i]->m;
    MeshGrad dx;
    for (int i = 0; i < 4; i++)
        if (free[i])
            dx[nodes[i]] = -(w[i]/nodes[i]->m)/inv_mass*n*d;
    return dx;
}

Tensor violation (Tensor value) {return max(-value, ZERO);}

Tensor IneqCon::energy (Tensor value) {
    Tensor v = violation(value);
    return stiff*v*v*v/::magic.repulsion_thickness/6;
}
Tensor IneqCon::energy_grad (Tensor value) {
    // cout << "IneqCon::energy_grad " << stiff << endl;
    // cout << "violation(value)" << violation(value) << endl;
    return -stiff*sq(violation(value))/::magic.repulsion_thickness/2;
}
Tensor IneqCon::energy_hess (Tensor value) {
    return stiff*violation(value)/::magic.repulsion_thickness;
}

MeshGrad IneqCon::friction (Tensor dt, MeshHess &jac) {
    if ((mu == 0).item<int>())
        return MeshGrad();
    Tensor fn = abs(energy_grad(value()));
    if ((fn == 0).item<int>())
        return MeshGrad();
    Tensor v = ZERO3;
    Tensor inv_mass = ZERO;
    for (int i = 0; i < 4; i++) {
        v = v + w[i]*nodes[i]->v;
        if (free[i])
            inv_mass = inv_mass + sq(w[i])/nodes[i]->m;
    }
    Tensor T = torch::eye(3,TNOPT) - ger(n,n);
    Tensor vt = norm(matmul(T,v));
    Tensor f_by_v = min(mu*fn/vt, 1/(dt*inv_mass));
    // double f_by_v = mu*fn/max(vt, 1e-1);
    MeshGrad force;
    for (int i = 0; i < 4; i++) {
        if (free[i]) {
            force[nodes[i]] = -w[i]*f_by_v*matmul(T,v);
            for (int j = 0; j < 4; j++) {
                if (free[j]) {
                    jac[make_pair(nodes[i],nodes[j])] = -w[i]*w[j]*f_by_v*T;
                }
            }
        }
    }
    return force;
}
