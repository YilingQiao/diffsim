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

#include "displaytesting.hpp"
#include "conf.hpp"
#include "display.hpp"
#include "dynamicremesh.hpp"
#include "geometry.hpp"
#include "io.hpp"
#include "opengl.hpp"
#include "plasticity.hpp"
#include "runphysics.hpp"
#include "util.hpp"
#include <cstdlib>
using namespace std;

#ifndef NO_OPENGL

static void recover_plasticity (Mesh &mesh) {
    for (int f = 0; f < mesh.faces.size(); f++)
        mesh.faces[f]->S_plastic = curvature<PS>(mesh.faces[f]);
}

static void remeshing_step (Cloth &cloth) {
    // copy old meshes
    Mesh old_mesh = deep_copy(cloth.mesh);
    // back up residuals
    vector<Residual> res = back_up_residuals(cloth.mesh);
    // remesh
    dynamic_remesh(cloth, vector<Plane>(), false);
    // restore residuals
    restore_residuals(cloth.mesh, old_mesh, res);
    // delete old meshes
    delete_mesh(old_mesh);
}

static void keyboard (unsigned char key, int x, int y) {
    unsigned char esc = 27, enter = 13;
    if (key == esc)
        exit(0);
    if (key == ' ') {
        recover_plasticity(sim.cloths[0].mesh);
    }
    if (key == enter) {
        remeshing_step(sim.cloths[0]);
    }
    redisplay();
}

void display_testing (const vector<string> &args) {
    if (args.size() == 0) {
        cout << "Placeholder command for interactively testing stuff." << endl;
        cout << "Edit src/displaytesting.cpp and make it do whatever you like!"
             << endl;
        exit(EXIT_FAILURE);
    }
    string conf = args[0];
    string meshfile = args[1];
    load_json(conf, sim);
    prepare(sim);
    load_obj(*sim.cloth_meshes[0], meshfile);
    GlutCallbacks cb;
    cb.keyboard = keyboard;
    run_glut(cb);
}

#else

void display_testing (const vector<string> &args) {opengl_fail();}

#endif // NO_OPENGL
