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

#include "geometry.hpp"
#include <cstdlib>

using namespace std;
using torch::Tensor;

Tensor signed_vf_distance (const Tensor &x,
                           const Tensor &y0, const Tensor &y1, const Tensor &y2,
                           Tensor *n, Tensor *w, double thres, bool &over) {
    return sub_signed_vf_distance(y0-x, y1-x, y2-x, n, w, thres, over);
}

Tensor sub_signed_vf_distance(const Tensor &y0, const Tensor &y1, const Tensor &y2,
                           Tensor *n, Tensor *w, double thres, bool &over) {
    Tensor _n; if (!n) n = &_n;
    Tensor _w[4]; if (!w) w = _w;
    *n = cross(y1-y0, y2-y0);
    w[0]=w[1]=w[2]=w[3]=ZERO;
    if ((norm2(*n) < 1e-6).item<int>()) {
        over = true;
        return infinity;
    }
    *n = normalize(*n);
    Tensor h = -dot(y0, *n);
    over = (abs(h) > thres).item<int>();
    if (over) return h;
    Tensor b0 = stp(y1, y2, *n),
           b1 = stp(y2, y0, *n),
           b2 = stp(y0, y1, *n);
    Tensor sum = 1/(b0 + b1 + b2);
    w[0] = ONE;
    w[1] = -b0*sum;
    w[2] = -b1*sum;
    w[3] = -b2*sum;
    return h;
}

Tensor signed_ee_distance (const Tensor &x0, const Tensor &x1,
                           const Tensor &y0, const Tensor &y1,
                           Tensor *n, Tensor *w, double thres, bool &over) {
    return sub_signed_ee_distance(x1-x0, y0-x0, y1-x0, y0-x1, y1-x1, y1-y0, n, w, thres, over);
}

Tensor sub_signed_ee_distance(const Tensor &x1mx0, const Tensor &y0mx0, const Tensor &y1mx0,
                           const Tensor &y0mx1, const Tensor &y1mx1, const Tensor &y1my0,
                           Tensor *n, Tensor *w, double thres, bool &over) {
    Tensor _n; if (!n) n = &_n;
    Tensor _w[4]; if (!w) w = _w;
    *n = cross(x1mx0, y1my0);
    w[0]=w[1]=w[2]=w[3]=ZERO;
    if ((norm2(*n) < 1e-6).item<int>()) {
        over = true;
        return infinity;
    }
    *n = normalize(*n);
    Tensor h = -dot(y0mx0, *n);
    over = (abs(h) > thres).item<int>();
    if (over) return h;
    // Tensor vec0 = torch::stack({y1mx1,y0mx0,y1mx1,y0mx0});
    // Tensor vec1 = torch::stack({y0mx1,y1mx0,y1mx0,y0mx1});
    // Tensor ab = tensordot(cross(vec0, vec1, 1), *n, {1},{0}).reshape({2,2});
    // ab = ab / sum(ab,1,true);
    // w[0] = ab[0][0];
    // w[1] = ab[0][1];
    // w[2] = -ab[1][0];
    // w[3] = -ab[1][1];
    Tensor a0 = stp(y1mx1, y0mx1, *n), a1 = stp(y0mx0, y1mx0, *n),
           b0 = stp(y1mx1, y1mx0, *n), b1 = stp(y0mx0, y0mx1, *n);
    Tensor suma = 1/(a0+a1), sumb = 1/(b0+b1);
    w[0] = a0*suma;
    w[1] = a1*suma;
    w[2] = -b0*sumb;
    w[3] = -b1*sumb;
    return h;
}

bool set_unsigned_ve_distance (const Tensor &x, const Tensor &y0, const Tensor &y1,
                               Tensor *_d, Tensor *_n,
                               Tensor *_wx, Tensor *_wy0, Tensor *_wy1) {
    Tensor t = clamp(dot(x-y0, y1-y0)/dot(y1-y0, y1-y0), 0., 1.);
    Tensor y = y0 + t*(y1-y0);
    Tensor d = norm(x-y);
    if ((d < *_d).item<int>()) {
        *_d = d;
        *_n = normalize(x-y);
        *_wx = ONE;
        *_wy0 = 1-t;
        *_wy1 = t;
        return true;
    }
    return false;
}

bool set_unsigned_vf_distance (const Tensor &x,
                               const Tensor &y0, const Tensor &y1, const Tensor &y2,
                               Tensor *_d, Tensor *_n,
                               Tensor *_wx,
                               Tensor *_wy0, Tensor *_wy1, Tensor *_wy2) {
    Tensor n = normalize(cross((y1-y0), (y2-y0)));
    Tensor d = abs(dot(x-y0, n));
    Tensor b0 = stp(y1-x, y2-x, n),
           b1 = stp(y2-x, y0-x, n),
           b2 = stp(y0-x, y1-x, n);
    if ((d < *_d).item<int>() && (b0 >= 0).item<int>() && (b1 >= 0).item<int>() && (b2 >= 0).item<int>()) {
        *_d = d;
        *_n = n;
        *_wx = ONE;
        *_wy0 = -b0/(b0 + b1 + b2);
        *_wy1 = -b1/(b0 + b1 + b2);
        *_wy2 = -b2/(b0 + b1 + b2);
        return true;
    }
    bool success = false;
    if ((b0 < 0).item<int>()
        && set_unsigned_ve_distance(x, y1, y2, _d, _n, _wx, _wy1, _wy2)) {
        success = true;
        *_wy0 = ZERO;
    }
    if ((b1 < 0).item<int>()
        && set_unsigned_ve_distance(x, y2, y0, _d, _n, _wx, _wy2, _wy0)) {
        success = true;
        *_wy1 = ZERO;
    }
    if ((b2 < 0).item<int>()
        && set_unsigned_ve_distance(x, y0, y1, _d, _n, _wx, _wy0, _wy1)) {
        success = true;
        *_wy2 = ZERO;
    }
    return success;
}

bool set_unsigned_ee_distance (const Tensor &x0, const Tensor &x1,
                               const Tensor &y0, const Tensor &y1,
                               Tensor *_d, Tensor *_n,
                               Tensor *_wx0, Tensor *_wx1,
                               Tensor *_wy0, Tensor *_wy1) {
    Tensor n = normalize(cross((x1-x0), (y1-y0)));
    Tensor d = abs(dot(x0-y0, n));
    Tensor a0 = stp(y1-x1, y0-x1, n), a1 = stp(y0-x0, y1-x0, n),
           b0 = stp(x0-y1, x1-y1, n), b1 = stp(x1-y0, x0-y0, n);
    if ((d < *_d).item<int>() && (a0 >= 0).item<int>() && (a1 >= 0).item<int>() && (b0 >= 0).item<int>() && (b1 >= 0).item<int>()) {
        *_d = d;
        *_n = n;
        *_wx0 = a0/(a0 + a1);
        *_wx1 = a1/(a0 + a1);
        *_wy0 = -b0/(b0 + b1);
        *_wy1 = -b1/(b0 + b1);
        return true;
    }
    bool success = false;
    if ((a0 < 0).item<int>()
        && set_unsigned_ve_distance(x1, y0, y1, _d, _n, _wx1, _wy0, _wy1)) {
        success = true;
        *_wx0 = ZERO;
    }
    if ((a1 < 0).item<int>()
        && set_unsigned_ve_distance(x0, y0, y1, _d, _n, _wx0, _wy0, _wy1)) {
        success = true;
        *_wx1 = ZERO;
    }
    if ((b0 < 0).item<int>()
        && set_unsigned_ve_distance(y1, x0, x1, _d, _n, _wy1, _wx0, _wx1)) {
        success = true;
        *_wy0 = ZERO;
        *_n = -*_n;
    }
    if ((b1 < 0).item<int>()
        && set_unsigned_ve_distance(y0, x0, x1, _d, _n, _wy0, _wx0, _wx1)) {
        success = true;
        *_wy1 = ZERO;
        *_n = -*_n;
    }
    return success;
}

Tensor unsigned_vf_distance (const Tensor &x,
                             const Tensor &y0, const Tensor &y1, const Tensor &y2,
                             Tensor *n, Tensor w[4]) {
    Tensor _n; if (!n) n = &_n;
    Tensor _w[4]; if (!w) w = _w;
    w[0]=w[1]=w[2]=w[3]=ZERO;
    *n=ZERO3;
    Tensor d = infinity;
    set_unsigned_vf_distance(x, y0, y1, y2, &d, n, &w[0], &w[1], &w[2], &w[3]);
    return d;
}

Tensor unsigned_ee_distance (const Tensor &x0, const Tensor &x1,
                             const Tensor &y0, const Tensor &y1,
                             Tensor *n, Tensor w[4]) {
    Tensor _n; if (!n) n = &_n;
    Tensor _w[4]; if (!w) w = _w;
    w[0]=w[1]=w[2]=w[3]=ZERO;
    *n=ZERO3;
    Tensor d = infinity;
    set_unsigned_ee_distance(x0, x1, y0, y1, &d, n, &w[0], &w[1], &w[2], &w[3]);
    return d;
}

Tensor get_barycentric_coords(const Tensor& point, const Face* f) {
    // Compute vectors        
    Tensor v0 = f->v[0]->u - f->v[2]->u;
    Tensor v1 = f->v[1]->u - f->v[2]->u;
    Tensor v2 = point - f->v[2]->u;
    // Compute dot products
    Tensor dot00 = dot(v0, v0);
    Tensor dot01 = dot(v0, v1);
    Tensor dot02 = dot(v0, v2);
    Tensor dot11 = dot(v1, v1);
    Tensor dot12 = dot(v1, v2);
    // Compute barycentric coordinates
    Tensor invDenom = 1. / (dot00 * dot11 - dot01 * dot01);
    Tensor u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    Tensor v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    return torch::stack({u,v,1-u-v});
}

// Is the point within the face?
// Adapted from http://www.blackpawn.com/texts/pointinpoly/default.html
bool is_inside(const Tensor& point, const Face* f) {
    Tensor bary = get_barycentric_coords(point, f);
    //printf("UV: %f, %f\n", u, v);
    // Check if point is in triangle
    // 10*epsilon: want to be robust for borders
    return ((bary[0] >= -10*EPSILON).item<int>() && (bary[1] >= -10*EPSILON).item<int>() && (bary[2] >= -100*EPSILON).item<int>());
}

// Gets the face that surrounds point u in material space
Face* get_enclosing_face(const Mesh& mesh, const Tensor& u,
                         Face *starting_face_hint) {
    for (int f = 0; f < mesh.faces.size(); f++)
        if (is_inside(u, mesh.faces[f]))
            return mesh.faces[f];
    return NULL;
}

template <> const Tensor &pos<PS> (const Node *node) {return node->y;}
template <> const Tensor &pos<WS> (const Node *node) {return node->x;}
template <> Tensor &pos<PS> (Node *node) {return node->y;}
template <> Tensor &pos<WS> (Node *node) {return node->x;}

template <Space s> Tensor nor (const Face *face) {
    const Tensor &x0 = pos<s>(face->v[0]->node),
               &x1 = pos<s>(face->v[1]->node),
               &x2 = pos<s>(face->v[2]->node);
    return normalize(cross(x1-x0, x2-x0));
}
template Tensor nor<PS> (const Face *face);
template Tensor nor<WS> (const Face *face);

Tensor unwrap_angle (Tensor theta, Tensor theta_ref) {
    if ((theta - theta_ref > M_PI).item<int>())
        theta -= 2*M_PI;
    if ((theta - theta_ref < -M_PI).item<int>())
        theta += 2*M_PI;
    return theta;
}

template <Space s> Tensor dihedral_angle (const Edge *edge) {
    // if (!hinge.edge[0] || !hinge.edge[1]) return 0;
    // const Edge *edge0 = hinge.edge[0], *edge1 = hinge.edge[1];
    // int s0 = hinge.s[0], s1 = hinge.s[1];
    if (!edge->adjf[0] || !edge->adjf[1])
        return ZERO;
    Tensor e = normalize(pos<s>(edge->n[0]) - pos<s>(edge->n[1]));
    if ((e==0).all().item<int>()) return ZERO;
    Tensor n0 = nor<s>(edge->adjf[0]), n1 = nor<s>(edge->adjf[1]);
    if ((n0==0).all().item<int>() || (n1==0).all().item<int>()) return ZERO;
    Tensor cosine = dot(n0, n1), sine = dot(e, cross(n0, n1));
    Tensor theta = atan2(sine, cosine);
    return unwrap_angle(theta, edge->reference_angle);
}
template Tensor dihedral_angle<PS> (const Edge *edge);
template Tensor dihedral_angle<WS> (const Edge *edge);

template <Space s> Tensor curvature (const Face *face) {
    Tensor S = torch::zeros({2,2},TNOPT);
    for (int e = 0; e < 3; e++) {
        const Edge *edge = face->adje[e];
        Tensor e_mat = face->v[PREV(e)]->u - face->v[NEXT(e)]->u,
             t_mat = perp(normalize(e_mat));
        Tensor theta = dihedral_angle<s>(face->adje[e]);
        S = S - 1/2.*theta*norm(e_mat)*ger(t_mat, t_mat);
    }
    S = S / face->a;
    return S;
}
template Tensor curvature<PS> (const Face *face);
template Tensor curvature<WS> (const Face *face);
