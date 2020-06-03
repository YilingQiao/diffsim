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

#include "nearobs.hpp"

#include "collisionutil.hpp"
#include "geometry.hpp"
#include "magic.hpp"
#include "simulation.hpp"
#include <vector>
using namespace std;
using torch::Tensor;

Tensor nearest_point (const Tensor &x, const vector<AccelStruct*> &accs,
                    Tensor dmin);

vector<Plane> nearest_obstacle_planes (const Mesh &mesh,
                                       const vector<Mesh*> &obs_meshes) {
    const Tensor dmin = 10*::magic.repulsion_thickness;
    vector<AccelStruct*> obs_accs = create_accel_structs(obs_meshes, false);
    vector<Plane> planes(mesh.nodes.size(), make_pair(ZERO3, ZERO3));
#pragma omp parallel for
    for (int n = 0; n < mesh.nodes.size(); n++) {
        Tensor x = mesh.nodes[n]->x;
        Tensor p = nearest_point(x, obs_accs, dmin);
        if ((p != x).any().item<int>())
            planes[n] = make_pair(p, normalize(x - p));
    }
    destroy_accel_structs(obs_accs);
    return planes;
}

struct NearPoint {
    Tensor d;
    Tensor x;
    NearPoint (Tensor d, const Tensor &x): d(d), x(x) {}
};

void update_nearest_point (const Tensor &x, BVHNode *node, NearPoint &p);

Tensor nearest_point (const Tensor &x, const vector<AccelStruct*> &accs,
                    Tensor dmin) {
    NearPoint p(dmin, x);
    for (int a = 0; a < accs.size(); a++)
        if (accs[a]->root)
            update_nearest_point(x, accs[a]->root, p);
    return p.x;
}

void update_nearest_point (const Tensor &x, const Face *face, NearPoint &p);

Tensor point_box_distance (const Tensor &x, const BOX &box);

void update_nearest_point (const Tensor &x, BVHNode *node, NearPoint &p) {
    if (node->isLeaf())
        update_nearest_point(x, node->getFace(), p);
    else {
        Tensor d = point_box_distance(x, node->_box);
        if ((d >= p.d).item<int>())
            return;
        update_nearest_point(x, node->getLeftChild(), p);
        update_nearest_point(x, node->getRightChild(), p);
    }
}

Tensor point_box_distance (const Tensor &x, const BOX &box) {
    Tensor xp = torch::stack({clamp(x[0], box._dist[0], box._dist[9]),
                   clamp(x[1], box._dist[1], box._dist[10]),
                   clamp(x[2], box._dist[2], box._dist[11])});
    return norm(x - xp);
}

void update_nearest_point (const Tensor &x, const Face *face, NearPoint &p) {
    Tensor n;
    Tensor w[4];
    w[0]=w[1]=w[2]=w[3]=ZERO;
    n=ZERO3;
    Tensor d = unsigned_vf_distance(x, face->v[0]->node->x, face->v[1]->node->x,
                                       face->v[2]->node->x, &n, w);
    if ((d < p.d).item<int>()) {
        p.d = d;
        p.x = -(w[1]*face->v[0]->node->x + w[2]*face->v[1]->node->x
              + w[3]*face->v[2]->node->x);
    }
}
