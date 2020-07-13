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

#ifndef TRANSFORMATION_HPP
#define TRANSFORMATION_HPP

#include "spline.hpp"
#include "vectors.hpp"
#include <iostream>
using torch::Tensor;

// Transform the mesh
struct Quaternion {
    Tensor s;
    Tensor v;
    Tensor rotate (const Tensor &point) const;
    static Quaternion from_axisangle(const Tensor &axis, Tensor angle);
    static Quaternion from_euler(const Tensor &euler);
    static Tensor to_euler(const Tensor &s, const Tensor &v);
    std::pair<Tensor, Tensor> to_axisangle() const;
    Quaternion operator+(const Quaternion& q) const;
    Quaternion operator-(const Quaternion& q) const;
    Quaternion operator-() const;
    Quaternion operator*(const Quaternion& q) const;
    Quaternion operator*(Tensor scalar) const;
    Quaternion operator/(Tensor scalar) const;
};

Quaternion normalize (const Quaternion &q);
Quaternion inverse(const Quaternion &q);
Tensor norm2(const Quaternion &q);
inline std::ostream &operator<< (std::ostream &out, const Quaternion &q) {out << "(" << q.s << ", " << q.v << ")"; return out;}

struct Transformation {
    Tensor translation;
    Tensor scale;
    Quaternion rotation;
    Transformation (Tensor factor=ONE);
    Tensor apply (const Tensor &point) const;
    Tensor apply_vec (const Tensor &vec) const;
    Transformation operator+(const Transformation& t) const;
    Transformation operator-(const Transformation& t) const;
    Transformation operator*(const Transformation& t) const;
    Transformation operator*(Tensor scalar) const;
    Transformation operator/(Tensor scalar) const;
    Tensor euler;  // yaw:z pitch:y roll:x 
};

Transformation identity ();
Transformation inverse(const Transformation &tr);
inline std::ostream &operator<< (std::ostream &out, const Transformation &t) {out << "(translation: " << t.translation << ", rotation: " << t.rotation << ", scale: " << t.scale << ")"; return out;}

typedef Spline<Transformation> Motion;
typedef std::pair<Transformation,Transformation> DTransformation;

void clean_up_quaternions (Motion &motion); // remove sign flips

Transformation get_trans (const Motion &motion, Tensor t);
DTransformation get_dtrans (const Motion &motion, Tensor t);
Tensor apply_dtrans (const DTransformation &dT, const Tensor &x0, Tensor *vel=NULL);
Tensor apply_dtrans_vec (const DTransformation &dT, const Tensor &v0);

#endif
