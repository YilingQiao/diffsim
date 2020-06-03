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

#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

#include "vectors.hpp"
#include "taucs.hpp"
#include <vector>
using torch::Tensor;

// Problems

struct NLOpt { // nonlinear optimization problem
    // minimize objective
    int nvar;
    virtual void initialize (Tensor &x) const = 0;
    virtual Tensor objective (const Tensor &x) const = 0;
    virtual void precompute (const Tensor &x) const {}
    virtual void gradient (const Tensor &x, Tensor &g) const = 0;
    virtual bool hessian (const Tensor &x, SpMat &H) const {
        return false; // should return true if implemented
    };
    virtual void finalize (const Tensor &x) const = 0;
};

struct NLConOpt { // nonlinear constrained optimization problem
    // minimize objective s.t. constraints = or <= 0
    int nvar, ncon;
    virtual void initialize (double *x) const = 0;
    virtual void precompute (const double *x) const {}
    virtual double objective (const double *x) const = 0;
    virtual void obj_grad (const double *x, double *grad) const = 0; // set
    virtual double constraint (const double *x, int j, int &sign) const = 0;
    virtual void con_grad (const double *x, int j, double factor,
                           double *grad) const = 0; // add factor*gradient
    virtual void finalize (const double *x) = 0;
};

// Algorithms

struct OptOptions {
    int _max_iter;
    double _eps_x, _eps_f, _eps_g;
    OptOptions (): _max_iter(100), _eps_x(1e-6), _eps_f(1e-12), _eps_g(1e-6) {}
    // Named parameter idiom
    // http://www.parashift.com/c++-faq-lite/named-parameter-idiom.html
    OptOptions &max_iter (int n) {_max_iter = n; return *this;}
    OptOptions &eps_x (double e) {_eps_x = e; return *this;}
    OptOptions &eps_f (double e) {_eps_f = e; return *this;}
    OptOptions &eps_g (double e) {_eps_g = e; return *this;}
    int max_iter () {return _max_iter;}
    double eps_x () {return _eps_x;}
    double eps_f () {return _eps_f;}
    double eps_g () {return _eps_g;}
};

void line_search_newtons_method (const NLOpt &problem,
                                 OptOptions opts=OptOptions(),
                                 bool verbose=false);

std::vector<double> augmented_lagrangian_method (NLConOpt &problem,
                                  OptOptions opts=OptOptions(),
                                  bool verbose=false);

// convenience functions for when optimization variables are Vec3-valued

#endif
