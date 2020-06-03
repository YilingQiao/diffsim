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

#include "handle.hpp"
#include "magic.hpp"
using namespace std;
using torch::Tensor;

static Tensor directions[3] = {torch::tensor({1,0,0},TNOPT), torch::tensor({0,1,0},TNOPT), torch::tensor({0,0,1},TNOPT)};

void add_position_constraints (const Node *node, const Tensor &x, Tensor stiff,
                               vector<Constraint*> &cons);

Transformation normalize (const Transformation &T) {
    Transformation T1 = T;
    T1.rotation = normalize(T1.rotation);
    return T1;
}

vector<Constraint*> NodeHandle::get_constraints (Tensor t) {
    Tensor s = strength(t);
    if ((s==0).item<int>())
        return vector<Constraint*>();
    if (!activated) {
        // handle just got started, fill in its original position
        x0 = motion ? inverse(normalize(motion->pos(t))).apply(node->x) : node->x;
        activated = true;
    }
    Tensor x = motion ? normalize(motion->pos(t)).apply(x0) : x0;
    vector<Constraint*> cons;
    add_position_constraints(node, x, s*::magic.handle_stiffness, cons);
    return cons;
}

vector<Constraint*> CircleHandle::get_constraints (Tensor t) {
    Tensor s = strength(t);
    if ((s==0).item<int>())
        return vector<Constraint*>();
    vector<Constraint*> cons;
    for (int n = 0; n < mesh->nodes.size(); n++) {
        Node *node = mesh->nodes[n];
        if (node->label != label)
            continue;
        Tensor theta = 2*M_PI*dot(node->verts[0]->u, u)/c;
        Tensor x = xc + (dx0*cos(theta) + dx1*sin(theta))*c/(2*M_PI);
        if (motion)
            x = motion->pos(t).apply(x);
        Tensor l = ZERO;
        for (int e = 0; e < node->adje.size(); e++) {
            const Edge *edge = node->adje[e];
            if (edge->n[0]->label != label || edge->n[1]->label != label)
                continue;
            l = l + edge->l;
        }
        add_position_constraints(node, x, s*::magic.handle_stiffness*l, cons);
    }
    return cons;
}

vector<Constraint*> GlueHandle::get_constraints (Tensor t) {
    Tensor s = strength(t);
    if ((s==0).item<int>())
        return vector<Constraint*>();
    vector<Constraint*> cons;
    for (int i = 0; i < 3; i++) {
        GlueCon *con = new GlueCon;
        con->nodes[0] = nodes[0];
        con->nodes[1] = nodes[1];
        con->n = directions[i];
        con->stiff = s*::magic.handle_stiffness;
        cons.push_back(con);
    }
    return cons;
}

void add_position_constraints (const Node *node, const Tensor &x, Tensor stiff,
                               vector<Constraint*> &cons) {
    for (int i = 0; i < 3; i++) {
        EqCon *con = new EqCon;
        con->node = (Node*)node;
        con->x = x;
        con->n = directions[i];
        con->stiff = stiff;
        cons.push_back(con);
    }
}
