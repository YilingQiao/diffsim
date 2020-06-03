#ifndef MORPH_HPP
#define MORPH_HPP

#include "mesh.hpp"
using torch::Tensor;

struct Morph {
    Mesh *mesh;
    std::vector<Mesh> targets;
    typedef Tensor Weights;
    Spline<Weights> weights;
    Spline<Tensor> log_stiffness;
    Tensor pos (Tensor t, const Tensor &u) const;
};

void apply (const Morph &morph, Tensor t);

#endif
