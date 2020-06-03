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

#ifndef MOT_PARSER_HPP
#define MOT_PARSER_HPP

#include "obstacle.hpp"
#include "util.hpp"
#include "vectors.hpp"
#include <string>
#include <iostream>
#include <vector>
using torch::Tensor;

std::vector<Motion> load_mot (const std::string &filename, double fps);

// I think I'm going to ignore all the rest down here

class mot_parser_exception {
public:
    mot_parser_exception(const std::string& error) : error(error) {}
    std::string error;
};

struct BodyFrame
{
    Tensor pos;
    Tensor orient;
};

typedef std::vector<BodyFrame> BodyFrameVector;
typedef std::vector<BodyFrameVector> BodyVector; 

BodyVector read_motion_file(const std::string& filename);
BodyVector read_motion_file(std::istream& istr);

void write_motion_file(BodyVector& bodies, const std::string& filename);
void write_motion_file(BodyVector& bodies, std::ostream& ostr);

BodyFrame& get_body_frame(BodyVector& bodies, size_t body_index, size_t frame);
BodyFrameVector& get_body_frames(BodyVector& bodies, size_t body_index);
//BodyFrameVector& get_body_frames(size_t body_index);
    
std::vector<Spline<Transformation> > mot_to_spline(std::string motion_file,
    const Transformation& tr, double fps, double start_time, double pause_time);

std::vector<Obstacle> mot_to_obs(std::string motion_file,
    const Transformation& tr, std::string obj_basename, double fps,
    double start_time, double pause_time);

#endif
