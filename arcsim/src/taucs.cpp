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

#include "taucs.hpp"
#include "timer.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <utility>
using namespace std;
using torch::Tensor;

extern "C" {
#include "taucs.h"
int taucs_linsolve (taucs_ccs_matrix* A, // input matrix
                    void** factorization, // an approximate inverse
                    int nrhs, // number of right-hand sides
                    void* X, // unknowns
                    void* B, // right-hand sides
                    char* options[], // options (what to do and how)
                    void* arguments[]); // option arguments
}

void add_submat (SpMat &A, int i, int j, const Tensor &Aij) {
  A(i,j) += Aij;//= A.slice(0,i*3,i*3+3).slice(1,j*3,j*3+3) + Aij;
}

ostream &operator<< (ostream &out, taucs_ccs_matrix *A) {
    out << "n: " << A->n << endl;
    out << "m: " << A->m << endl;
    out << "flags: " << A->flags << endl;
    out << "colptr: ";
    for (int i = 0; i <= A->n; i++)
        out << (i==0?"":", ") << A->colptr[i];
    out << endl;
    out << "rowind: ";
    for (int j = 0; j <= A->colptr[A->n]; j++)
        out << (j==0?"":", ") << A->rowind[j];
    out << endl;
    out << "values.d: ";
    for (int j = 0; j <= A->colptr[A->n]; j++)
        out << (j==0?"":", ") << A->values.d[j];
    out << endl;
    return out;
}
std::ostream &operator<< (std::ostream &out, const SpMat &A) {
    out << "[";
    for (int i = 0; i < A.m; i++) {
        const SpVec &row = A.rows[i];
        for (int jj = 0; jj < row.indices.size(); jj++) {
            int j = row.indices[jj];
            const Tensor &aij = row.entries[jj];
            out << (i==0 && jj==0 ? "" : ", ") << "(" << i << "," << j
                << "): " << aij;
        }
    }
    out << "]";
    return out;
}

taucs_ccs_matrix *sparse_to_taucs (const Tensor &As, vector<pair<int, int> > &indices, int n) {
    // assumption: A is square and symmetric
    int nnz = 0;
    for (auto p : indices) {
        int i = p.first, j = p.second;
        if (j < i)
            continue;
        nnz += (j==i) ? 6 : 9;
    }
    taucs_ccs_matrix *At = taucs_ccs_create
        (n,n, nnz, TAUCS_DOUBLE | TAUCS_SYMMETRIC | TAUCS_LOWER);
    auto foo_a = As.accessor<double,3>();
    int pos = 0, indi = 0;
    for (int i = 0; i < n/3; i++) {
        for (int k = 0; k < 3; k++) {
            At->colptr[i*3+k] = pos;
            while (indi < indices.size() && indices[indi].first < i)
                ++indi;
            if (indi >= indices.size() || indices[indi].first > i)
                continue;
            for (int ind = indi; ind < indices.size(); ++ind) {
                if (indices[ind].first > i)
                    break;
                int j = indices[ind].second;
                if (j < i)
                    continue;
                for (int l = (i==j)? k : 0; l < 3; ++l) {
                    At->rowind[pos] = j*3+l;
                    At->values.d[pos] = foo_a[ind][k][l];
                    pos++;
                }
            }
        }
    }
    At->colptr[n] = pos;
    return At;
}

Tensor taucs_linear_solve_forward (Tensor A, Tensor b, vector<pair<int, int> > indices) {
    int n = b.size(0);
    taucs_ccs_matrix *Ataucs = sparse_to_taucs(A, indices, n);
    //cout << Ataucs << endl;
    vector<double> x(n);
    char *options[] = {(char*)"taucs.factor.LLT=true",(char*)"taucs.factor.ordering=amd", NULL};
    int retval = taucs_linsolve(Ataucs, NULL, 1, &x[0], b.data<double>(), options, NULL);
    if (retval != TAUCS_SUCCESS) {
        cerr << "Error: TAUCS failed with return value " << retval << endl;
        exit(EXIT_FAILURE);
    }
    taucs_ccs_free(Ataucs);
    return torch::tensor(x, TNOPT);
}
vector<Tensor> taucs_linear_solve_backward (Tensor dldz, vector<pair<int, int> > indices, Tensor ans, Tensor A, Tensor b) {
    int n = b.size(0);
    taucs_ccs_matrix *Ataucs = sparse_to_taucs(A, indices, n);
    vector<double> x(n);
    char *options[] = {(char*)"taucs.factor.LLT=true",(char*)"taucs.factor.ordering=amd", NULL};
    int retval = taucs_linsolve(Ataucs, NULL, 1, &x[0], dldz.data<double>(), options, NULL);
    if (retval != TAUCS_SUCCESS) {
        cerr << "Error: TAUCS failed with return value " << retval << endl;
        exit(EXIT_FAILURE);
    }
    taucs_ccs_free(Ataucs);
    Tensor dx = torch::tensor(x, TNOPT);
    vector<Tensor> ret;
    Tensor dxtmp = dx.reshape({-1,3});
    ans = ans.reshape({-1,3});
    for (auto p : indices) {
        int row = p.first, col = p.second;
        ret.push_back(ger(dxtmp[row], ans[col]));
    }
    Tensor dlda = -torch::stack(ret);
    return {dlda, dx};
}

Tensor taucs_linear_solve (const SpMat &A, const Tensor &b) {
  // taucs_logfile("stdout");
  vector<Tensor> atmp;
  vector<pair<int,int> > indices;
  for (int l = 0; l < A.m; ++l) {
    const SpVec &row = A.rows[l];
    for (int k = 0; k < row.indices.size(); ++k) {
      atmp.push_back(row.entries[k]);
      indices.push_back(make_pair(l, row.indices[k]));
    }
  }
  // for (int i = 0; i < indices.size(); ++i) {
  //   cout << atmp[i] << endl;
  //   cout << indices[i].first << " " << indices[i].second << endl;
  // }
  py::object func = py::module::import("taucs_py").attr("taucs_linear_solve");
  Tensor ans = func(torch::stack(atmp), b, indices).cast<Tensor>();
  // Tensor ans = taucs_linear_solve_forward(torch::stack(atmp), b, indices);
  return ans;
}

