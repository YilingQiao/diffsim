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

#include "transformation.hpp"

using namespace std;
using torch::Tensor;
 
Transformation identity () {
    return Transformation();
}

Transformation inverse(const Transformation &tr) {
    Transformation in;
    in.scale = 1. / tr.scale;
    in.rotation = inverse(tr.rotation);
    in.translation = ZERO3 -
        in.rotation.rotate(
            in.scale * (
                tr.translation
            )
        );
    return in;
}

Quaternion inverse(const Quaternion &q) {
    Quaternion in;
    Tensor divisor = norm2(q);
    in.s = q.s / divisor;
    in.v = -q.v / divisor;
    return in;
}

Quaternion Quaternion::from_axisangle(const Tensor &axis, Tensor angle) {
    Quaternion q;
    if ((angle == 0).item<int>()) {
        q.s = ONE;
        q.v = ZERO3;
    } else {
        q.s = cos(angle/2);
        q.v = sin(angle/2)*normalize(axis);
    }
    return q;
}
 


Quaternion Quaternion::from_euler(const Tensor &euler) {
    // yaw (Z), pitch (Y), roll (X)
    Tensor yaw=euler[2], pitch=euler[1], roll=euler[0];
    Quaternion q;

    Tensor cy = torch::cos(yaw * 0.5);
    Tensor sy = torch::sin(yaw * 0.5);
    Tensor cp = torch::cos(pitch * 0.5);
    Tensor sp = torch::sin(pitch * 0.5);
    Tensor cr = torch::cos(roll * 0.5);
    Tensor sr = torch::sin(roll * 0.5);

    q.s   = cy * cp * cr + sy * sp * sr;

    q.v    = ZERO3;
    q.v[0] = cy * cp * sr - sy * sp * cr;
    q.v[1] = sy * cp * sr + cy * sp * cr;
    q.v[2] = sy * cp * cr - cy * sp * sr;

    return q;
}


Tensor Quaternion::to_euler(const Tensor &s, const Tensor &v) {
    Tensor w = s;
    Tensor x=v[0], y=v[1], z=v[2];

    Tensor yaw, pitch, roll;

    // roll (x-axis rotation)
    Tensor sinr_cosp = 2 * (w * x + y * z);
    Tensor cosr_cosp = 1 - 2 * (x * x + y * y);
    roll = torch::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    Tensor sinp = 2 * (w * y - z * x);
    pitch = torch::where(torch::abs(sinp) >= ONE, // use 90 degrees if out of range
                         torch::where(sinp > ZERO, DT_PI/2, -DT_PI/2), 
                         torch::asin(sinp));
    // yaw (z-axis rotation)
    Tensor siny_cosp = 2 * (w * z + x * y);
    Tensor cosy_cosp = 1 - 2 * (y * y + z * z);
    yaw = torch::atan2(siny_cosp, cosy_cosp);

    Tensor euler = ZERO3;
    euler[2] = yaw;
    euler[1] = pitch;
    euler[0] = roll;

    return euler;
}


pair<Tensor, Tensor> Quaternion::to_axisangle() const {
    Tensor angle = 2 * acos(s);
    Tensor axis;
    if((angle == 0).item<int>()) {
        axis = torch::ones({3},TNOPT);
    } else {
        axis = v / sqrt(1.0-s*s);
    }
    return pair<Tensor, Tensor>(axis, angle);
}

Transformation::Transformation(Tensor factor) {
    translation = ZERO3;
    scale = factor;
    rotation = Quaternion::from_axisangle(torch::ones({3},TNOPT), ZERO)*factor;
}

Transformation Transformation::operator-(const Transformation& other) const {
    Transformation t;
    t.scale = this->scale - other.scale;
    t.translation = this->translation - other.translation;
    t.rotation = this->rotation - other.rotation;
    return t;
}

Transformation Transformation::operator+(const Transformation& other) const {
    Transformation t;
    t.scale = this->scale + other.scale;
    t.translation = this->translation + other.translation;
    t.rotation = this->rotation + other.rotation;
    return t;
}

Transformation Transformation::operator*(const Transformation& other) const {
    Transformation t;
    t.scale = this->scale * other.scale;
    t.translation = this->translation + 
                    this->rotation.rotate(other.translation * this->scale);
    t.rotation = this->rotation * other.rotation;
    return t;
}

Transformation Transformation::operator*(Tensor s) const {
    Transformation t;
    t.scale = this->scale * s;
    t.translation = this->translation * s;
    t.rotation = this->rotation * s;
    return t;
}

Transformation Transformation::operator/(Tensor s) const {
    return (*this)*(1./s);
}

Quaternion Quaternion::operator+(const Quaternion& other) const {
    Quaternion q;
    q.v = this->v + other.v;
    q.s = this->s + other.s;
    return q;
}

Quaternion Quaternion::operator-(const Quaternion& other) const {
    Quaternion q;
    q.v = this->v - other.v;
    q.s = this->s - other.s;
    return q;
}

Quaternion Quaternion::operator-() const {
    Quaternion q;
    q.v = -this->v;
    q.s = -this->s;
    return q;
}

Quaternion Quaternion::operator*(const Quaternion& other) const {
    Quaternion q;
    q.v = (this->s * other.v) + (other.s * this->v) +
               cross(this->v, other.v);
    q.s = (this->s * other.s) - dot(this->v, other.v);
    return q;
}

Quaternion Quaternion::operator*(Tensor s) const {
    Quaternion q;
    q.v = this->v * s;
    q.s = this->s * s;
    return q;
}

Quaternion Quaternion::operator/(Tensor s) const {
    return (*this)*(1./s);
}

Tensor Quaternion::rotate (const Tensor &x) const {
    return x*(sq(s) - dot(v,v)) +
           2.*v*dot(v,x) + 2.*cross(v,x)*s;
}

Tensor Transformation::apply (const Tensor &x) const {
    return translation + scale*rotation.rotate(x);
}

Tensor Transformation::apply_vec (const Tensor &v) const {
    return rotation.rotate(v);
}

Tensor norm2(const Quaternion &q) {
    return sq(q.s) + norm2(q.v);
}

Quaternion normalize (const Quaternion &q) {
    Tensor norm = sqrt(norm2(q));
    Quaternion p;
    p.s = q.s/norm;
    p.v = q.v/norm;
    return p;
}

void clean_up_quaternions (Motion &motion) {
    for (int p = 1; p < motion.points.size(); p++) {
        const Quaternion &q0 = motion.points[p-1].x.rotation;
        Quaternion &q1 = motion.points[p].x.rotation;
        Tensor d = dot(q0.v, q1.v) + q0.s*q1.s;
        if ((d < 0).item<int>())
            q1 = -q1;
    }
}

Transformation get_trans (const Motion &motion, Tensor t) {
    Transformation T = motion.pos(t);
    T.rotation = normalize(T.rotation);
    return T;
}

DTransformation get_dtrans (const Motion &motion, Tensor t) {
    Transformation T = motion.pos(t), dT = motion.vel(t);
    Quaternion q = T.rotation, dq = dT.rotation;
    Tensor qq = sq(q.s) + norm2(q.v),
           qdq = q.s*dq.s + dot(q.v, dq.v);
    Tensor normq = sqrt(qq);
    T.rotation = q/normq;
    dT.rotation = dq/normq - q/normq*qdq/qq;
    return make_pair(T, dT);
}

Tensor apply_dtrans (const DTransformation &dtrans, const Tensor &x0, Tensor *vel) {
    const Transformation &T = dtrans.first, &dT = dtrans.second;
    Tensor rot = T.rotation.rotate(x0);
    Tensor x = T.translation+T.scale*rot;
    if (vel) {
        Tensor w = 2.*(dT.rotation*inverse(T.rotation)).v;
        *vel = dT.translation + dT.scale*rot
             + T.scale*cross(w, rot);
    }
    return x;
}

Tensor apply_dtrans_vec (const DTransformation &dtrans, const Tensor &v0) {
    return dtrans.first.apply_vec(v0);
}
