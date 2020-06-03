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

#ifndef UTIL_HPP
#define UTIL_HPP

#include <string> // aa: win
#include "mesh.hpp"
#include "spline.hpp"
#include <iostream>
#include <string>
#include <vector>
using torch::Tensor;

#define EPSILON		1e-7f

// i+1 and i-1 modulo 3
// This way of computing it tends to be faster than using %
#define NEXT(i) ((i)<2 ? (i)+1 : (i)-2)
#define PREV(i) ((i)>0 ? (i)-1 : (i)+2)

typedef unsigned int uint;

struct Transformation;

// Quick-and easy statistics

// sprintf for std::strings

std::string stringf (const std::string &format, ...);

// Easy reporting of vertices and faces

std::ostream &operator<< (std::ostream &out, const Vert *vert);
std::ostream &operator<< (std::ostream &out, const Node *node);
std::ostream &operator<< (std::ostream &out, const Edge *edge);
std::ostream &operator<< (std::ostream &out, const Face *face);

// Math utilities

extern const Tensor infinity;



int solve_quadratic (Tensor a, Tensor b, Tensor c, Tensor x[2]);

Tensor solve_cubic(Tensor a, Tensor b, Tensor c, Tensor d);

// Find, replace, and all that jazz

template <typename T> inline int find (const T *x, T* const *xs, int n=3) {
    for (int i = 0; i < n; i++) if (xs[i] == x) return i; return -1;}

template <typename T> inline int find (const T &x, const T *xs, int n=3) {
    for (int i = 0; i < n; i++) if (xs[i] == x) return i; return -1;}

template <typename T> inline int find (const T &x, const std::vector<T> &xs) {
    for (int i = 0; i < xs.size(); i++) if (xs[i] == x) return i; return -1;}

template <typename T> inline bool is_in (const T *x, T* const *xs, int n=3) {
    return find(x, xs, n) != -1;}

template <typename T> inline bool is_in (const T &x, const T *xs, int n=3) {
    return find(x, xs, n) != -1;}

template <typename T> inline bool is_in (const T &x, const std::vector<T> &xs) {
    return find(x, xs) != -1;}

template <typename T> inline void include (const T &x, std::vector<T> &xs) {
    if (!is_in(x, xs)) xs.push_back(x);}

template <typename T> inline void remove (int i, std::vector<T> &xs) {
    xs[i] = xs.back(); xs.pop_back();}

template <typename T> inline void exclude (const T &x, std::vector<T> &xs) {
    int i = find(x, xs); if (i != -1) remove(i, xs);}

template <typename T> inline void replace (const T &v0, const T &v1, T vs[3]) {
    int i = find(v0, vs); if (i != -1) vs[i] = v1;}

template <typename T>
inline void replace (const T &x0, const T &x1, std::vector<T> &xs) {
    int i = find(x0, xs); if (i != -1) xs[i] = x1;}

template <typename T>
inline bool subset (const std::vector<T> &xs, const std::vector<T> &ys) {
    for (int i = 0; i < xs.size(); i++) if (!is_in(xs[i], ys)) return false;
    return true;
}

template <typename T>
inline void append (std::vector<T> &xs, const std::vector<T> &ys) {
    xs.insert(xs.end(), ys.begin(), ys.end());}

// Comparisons on vectors

// Mesh utilities

bool is_seam_or_boundary (const Vert *v);
bool is_seam_or_boundary (const Node *n);
bool is_seam_or_boundary (const Edge *e);
bool is_seam_or_boundary (const Face *f);

// Debugging

void debug_save_mesh (const Mesh &mesh, const std::string &name, int n=-1);
void debug_save_meshes (const std::vector<Mesh*> &meshes,
                        const std::string &name, int n=-1);

template <typename T>
std::ostream &operator<< (std::ostream &out, const std::vector<T> &v) {
    out << "[";
    for (int i = 0; i < v.size(); i++)
        out << (i==0 ? "" : ", ") << v[i];
    out << "]";
    return out;
}

#define ECHO(x) std::cout << #x << std::endl

#define REPORT(x) std::cout << #x << " = " << (x) << std::endl

#define REPORT_ARRAY(x,n) std::cout << #x << "[" << #n << "] = " << vector<double>(&(x)[0], &(x)[n]) << std::endl

#endif
