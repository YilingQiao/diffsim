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

#include "mot_parser.hpp"

#include "io.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using torch::Tensor;

std::vector<Motion> load_mot (const std::string &filename, double fps) {
    return mot_to_spline(filename, identity(), fps, 0, 0);
}

bool is_all_whitespace(const string& empty) {
    for (uint i = 0; i < empty.size(); ++i)
        if (isprint(empty[i]))
            return false;
    
    return true;
}

size_t num_frames(BodyVector &bodies) {
    if (bodies.size() == 0)
        return 0;
    else
        return bodies[0].size(); 
}

size_t num_bodies(BodyVector &bodies) {
    return bodies.size();
}

void append_frame(BodyVector &bodies, size_t body_index, const BodyFrame& bf) {
    if (body_index < 0 || body_index >= num_bodies(bodies)) {
        throw mot_parser_exception("cannot append frame: index out of bounds");
    }

    // TODO: add exception check here
    BodyFrameVector& bfv = bodies[body_index];
    bfv.push_back(bf);
}

void resize(BodyVector &bodies, size_t nbodies, size_t nframes) {
    bodies.resize(nbodies);

    for (size_t i = 0; i < nbodies; ++i) {
        bodies[i].resize(nframes);
    }
}


BodyVector read_motion_file(const string& filename) {
    ifstream fin(filename.c_str());
    if (!fin)
        throw mot_parser_exception(string("cannot read mot") + filename);
    BodyVector bodies = read_motion_file(fin);
    fin.close();
    return bodies;
}

BodyVector read_motion_file(std::istream& istr) {
    BodyVector bodies;
    stringstream ss;
    string stmp;
    int bidx = 0;
    int num_frames = 0;
    vector<double> dtmp(3);
    
    getline(istr, stmp);
    ss << stmp;
    
    ss >> stmp;
    ss >> num_frames;
    ss.clear();
    
    while (getline(istr, stmp)) 
    {
        if (is_all_whitespace(stmp))
            continue;
        
        bodies.resize(bidx + 1);
        bodies[bidx].resize(num_frames);
        
        // get position for each frame
        for (int i = 0; i < num_frames; ++i) 
        {
            getline(istr, stmp);
            ss.clear();
            ss << stmp;
           
            for (int j = 0; j < 3; ++j) {
                ss >> dtmp[j];
            }
            bodies[bidx][i].pos = torch::tensor(dtmp,TNOPT);
        }
        
        // skip preceeding empty lines and header
        while (getline(istr, stmp) && is_all_whitespace(stmp));
        
        // get orientation for each frame
        for (int i = 0; i < num_frames; ++i)
        {
            getline(istr, stmp);
            ss.clear();
            ss << stmp;
          
            for (int j = 0; j < 4; ++j) {
                ss >> dtmp[j];
            }
            bodies[bidx][i].orient = torch::tensor(dtmp,TNOPT);
            
            /*
            for (int j = 0; j < 4; ++j) {
                cout << bodies[bidx][i].orient[j] << " ";
            }
            cout << endl;
            */
        }
        
        ++bidx;
    }
    return bodies;
}

void write_motion_file(BodyVector &bodies, const string& filename) {
    ofstream fout(filename.c_str());
    
    write_motion_file(bodies, fout);
    
    fout.close();
}

void write_motion_file(BodyVector &bodies, ostream& ostr) {
    ostr << "NumFrames: " << num_frames(bodies) << endl;
   
    for (size_t i = 0; i < num_bodies(bodies); ++i)
    {
        ostr << "body[" << i << "] position" << endl;
        for (size_t j = 0; j < num_frames(bodies); ++j)
        {
            const BodyFrame& f = get_body_frame(bodies, i, j);
            ostr << f.pos[0].item<double>() << " " << f.pos[1].item<double>() << " " << f.pos[2].item<double>() << endl;
        }
        
        ostr << endl;
        
        ostr << "body[" << i << "] orientation" << endl;
        for (size_t j = 0; j < num_frames(bodies); ++j)
        {
            const BodyFrame& f = get_body_frame(bodies, i, j);
            ostr << f.orient[0].item<double>() << " " << f.orient[1].item<double>() << " " <<
                f.orient[2].item<double>() << " " << f.orient[3].item<double>() << endl;
        }
        
        ostr << endl;
    }
}

BodyFrame& get_body_frame(BodyVector &bodies, size_t body_index, size_t frame) {
    BodyFrameVector& bfv = get_body_frames(bodies, body_index);
    
    if (frame < 0 || frame >= bfv.size())
        throw mot_parser_exception("frame index out of bounds");
    
    return bfv[frame];
}

BodyFrameVector& get_body_frames(BodyVector &bodies, size_t body_index) {
    if (body_index < 0 || body_index >= bodies.size())
        throw mot_parser_exception("body index out of bounds");

    return bodies[body_index];
}

Transformation bodyframe_to_transformation(const BodyFrame& bodyFrame) {
    Transformation tr = identity();
    tr.rotation.s = bodyFrame.orient[0];
    tr.rotation.v = bodyFrame.orient.slice(0,0,3);
    tr.translation = bodyFrame.pos;
    return tr;
}

vector<vector<Transformation> > body_vector_to_transforms(BodyVector& bodies) {
    vector<vector<Transformation> > body_transforms(bodies.size());
    for(int body_i = 0; body_i < bodies.size(); ++body_i) {
        const BodyFrameVector& curr_body = bodies[body_i];
        // Initialize new vector
        body_transforms[body_i] = vector<Transformation>(curr_body.size());
        for(int frame_i = 0; frame_i < curr_body.size(); ++frame_i) {
            Transformation tr = bodyframe_to_transformation(curr_body[frame_i]);
            body_transforms[body_i][frame_i] = tr;
        }
    }
    return body_transforms;
}

vector<vector<Transformation> > mot_to_transforms(string motion_file) {
    // Parse the motion file
    BodyVector bodies = read_motion_file(motion_file);
    vector<vector<Transformation> > body_transforms =
                                    body_vector_to_transforms(bodies);
    return body_transforms;
}

Spline<Transformation> build_cubic_spline(
        const vector<Transformation> &transforms, double start_time,
        double fps) {
    int num_frames = transforms.size();
    Spline<Transformation> transform_spline;
    transform_spline.points = vector<Spline<Transformation>::Point>(num_frames);
    // Add one frame at a time
    for(int frame_i = 0; frame_i < num_frames; ++frame_i) {
        Spline<Transformation>::Point point;
        point.t = ONE*(frame_i / fps) - start_time;
        point.x = transforms[frame_i];
        // velocity is filled in below
        transform_spline.points[frame_i] = point;
    }
    for(int frame_i = 0; frame_i < num_frames; ++frame_i)
        fill_in_velocity(transform_spline, frame_i);
    return transform_spline;
}

vector<Spline<Transformation> > mot_to_spline(string motion_file, const Transformation& tr,
                           double fps, double start_time, double pause_time) {
    // Parse the motion file
    vector<vector<Transformation> > body_transforms =
                                    mot_to_transforms(motion_file);
    int num_body_parts = body_transforms.size();
    int num_transforms = body_transforms[0].size();
    // Transform all transforms by the base transformation
    for(int body_i=0; body_i < num_body_parts; ++body_i) {
        for(int spline_i=0; spline_i < num_transforms; ++spline_i) {
            body_transforms[body_i][spline_i] =
                tr * body_transforms[body_i][spline_i];
        }
    }
    // Turn it into a vector of splines
    vector<Spline<Transformation> > splines(num_body_parts);
    for(int i=0; i<num_body_parts; ++i) {
        splines[i] = build_cubic_spline(body_transforms[i], start_time, fps);
    }
    return splines;
}
