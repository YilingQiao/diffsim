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

#include "optimization.hpp"

#include "taucs.hpp"
#include "util.hpp"

using namespace std;
using torch::Tensor;

static bool verbose;

Tensor line_search (const Tensor &x0, const Tensor &p,
                    const NLOpt &problem, Tensor f, const Tensor &g);

void line_search_newtons_method (const NLOpt &problem, OptOptions opt,
                                 bool verbose) {
    ::verbose = verbose;
    int n = problem.nvar;
    Tensor x = torch::zeros({n},TNOPT), g = torch::zeros({n},TNOPT);
    SpMat H(n/3,n/3);// = torch::zeros({n,n},TNOPT);
    problem.initialize(x);
    Tensor f_old = infinity;
    int iter;
    for (iter = 0; iter < opt.max_iter(); iter++) {
        problem.precompute(x);
        Tensor f = problem.objective(x);
        if (verbose)
            REPORT(f);
        if ((f_old - f < opt.eps_f()).item<int>())
            break;
        f_old = f;
        problem.gradient(x, g);
        if (verbose)
            REPORT(norm(g));
        if ((norm(g) < opt.eps_g()).item<int>())
            break;
        if (!problem.hessian(x, H)) {
            cerr << "Can't run Newton's method if Hessian of objective "
                 << "is not available!" << endl;
            exit(1);
        }
        Tensor p = taucs_linear_solve(H, g);
        // for (int i = 0; i < n; ++i)cout << g[i].item<double>() << endl;
        if (verbose)
            REPORT(norm(p));
        p = -p;
        Tensor a = line_search(x, p, problem, f, g);
//        cout << a.item<double>() << endl;
//for (int i = 0; i < 3; ++i)
//cout << g[5*3+i].item<double>() << ",";cout << endl;
        x = x + a * p;
        if ((a*norm(p) < opt.eps_x()).item<int>())
            break;
    }
    if (verbose)
        REPORT(iter);
    problem.finalize(x);
}

inline Tensor cb (Tensor x) {return x*x*x;}

Tensor line_search (const Tensor &x0, const Tensor &p,
                    const NLOpt &problem, Tensor f0, const Tensor &g) {
    Tensor c = ONE*1e-3; // sufficient decrease parameter
    Tensor a = ONE;
    int n = problem.nvar;
    Tensor x = torch::zeros({n},TNOPT);
    Tensor g0 = dot(g, p);
    if (::verbose)
        REPORT(g0);
    if ((abs(g0) < 1e-12).item<int>())
        return ZERO;
    Tensor a_prev = ZERO;
    Tensor f_prev = f0;
    while (true) {
        x = x0 + a * p;
        // problem.precompute(&x[0]);
        Tensor f = problem.objective(x);
        if (::verbose) {
            REPORT(a);
            REPORT(f);
        }
        // cout << "\t"<<a.item<double>() << endl;
        if ((f <= f0 + c*a*g0).item<int>())
            break;
        Tensor a_next;
        if ((a_prev == ZERO).item<int>())
            a_next = -g0*sq(a)/(2*(f - f0 - g0*a)); // minimize quadratic fit
        else {
            // minimize cubic fit to f0, g0, f_prev, f
            Tensor b = matmul(torch::stack({sq(a_prev), -sq(a),-cb(a_prev),cb(a)}).reshape({2,2})
                     , torch::stack({f-f0-g0*a, f_prev-f0-g0*a_prev}))
                     / (sq(a)*sq(a_prev)*(a-a_prev));
            Tensor a_sol[2];
            solve_quadratic(3*b[0], 2*b[1], g0, a_sol);
            a_next = (a_sol[0] > 0).item<int>() ? a_sol[0] : a_sol[1];
        }
        if ((a_next < a*0.1).item<int>() || (a_next > a*0.9).item<int>())
            a_next = a/2;
        a_prev = a;
        f_prev = f;
        a = a_next;
    }
    return a;
}
