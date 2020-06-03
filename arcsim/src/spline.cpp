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

#include "spline.hpp"
#include "util.hpp"

using namespace std;
using torch::Tensor;

// binary search, returns keyframe immediately *after* given time
// range of output: 0 to a.keyfs.size() inclusive
template<typename T>
static int find (const Spline<T> &s, Tensor t) {
    int l = 0, u = s.points.size();
    while (l != u) {
         int m = (l + u)/2;
         if ((t < s.points[m].t).template item<int>()) u = m;
         else l = m + 1;
    }
    return l; // which is equal to u
}

template<typename T>
T Spline<T>::pos (Tensor t) const {
    int i = find(*this, t);
    if (i == 0) {
         const Point &p1 = points[i];
         return p1.x;
    } else if (i == points.size()) {
         const Point &p0 = points[i-1];
         return p0.x;
    } else {
         const Point &p0 = points[i-1], &p1 = points[i];
         Tensor s = (t - p0.t)/(p1.t - p0.t), s2 = s*s, s3 = s2*s;
         return p0.x*(2*s3 - 3*s2 + 1) + p1.x*(-2*s3 + 3*s2)
             + (p0.v*(s3 - 2*s2 + s) + p1.v*(s3 - s2))*(p1.t - p0.t);
    }
}

template <typename T>
T Spline<T>::vel (Tensor t) const {
    int i = find(*this, t);
    if (i == 0 || i == points.size()) {
        return points[0].x-points[0].x;
    } else {
        const Point &p0 = points[i-1], &p1 = points[i];
        Tensor s = (t - p0.t)/(p1.t - p0.t), s2 = s*s;
        return (p0.x*(6*s2 - 6*s) + p1.x*(-6*s2 + 6*s))/(p1.t - p0.t)
            + p0.v*(3*s2 - 4*s + 1) + p1.v*(3*s2 - 2*s);
    }
}

template class Spline<Tensor>;
template class Spline<Transformation>;
