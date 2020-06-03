#include "morph.hpp"

#include "geometry.hpp"
using namespace std;
using torch::Tensor;

Tensor blend (const vector<Mesh> &targets, const Tensor &w,
            const Tensor &u) {
    Tensor x = ZERO3;
    for (int m = 0; m < targets.size(); m++) {
        if ((w[m] == ZERO).item<int>())
            continue;
        Face *face = get_enclosing_face(targets[m], u);
        if (!face)
            continue;
        Tensor b = get_barycentric_coords(u, face);
        x = x + w[m]*(b[0]*face->v[0]->node->x
                   + b[1]*face->v[1]->node->x
                   + b[2]*face->v[2]->node->x);
    }
    return x;
}

Tensor Morph::pos (Tensor t, const Tensor &u) const {
    return blend(targets, weights.pos(t), u);
}

void apply (const Morph &morph, Tensor t) {
    for (int n = 0; n < morph.mesh->nodes.size(); n++) {
        Node *node = morph.mesh->nodes[n];
        Tensor x = ZERO3;
        for (int v = 0; v < node->verts.size(); v++)
            x = x + morph.pos(t, node->verts[v]->u);
        node->x = x/(double)node->verts.size();
    }
}
