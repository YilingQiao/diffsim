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

#include "displayphysics.hpp"

#include "display.hpp"
#include "io.hpp"
#include "opengl.hpp"
#include "misc.hpp"
#include "runphysics.hpp"
#include "simulation.hpp"
#include "timer.hpp"
#include "util.hpp"

#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
using namespace std;

#ifndef NO_OPENGL

extern string outprefix;
extern fstream timingfile;

static bool running = false;

static void idle () {
    if (!::running)
        return;
    sim_step();
    redisplay();
}

extern void zoom (bool in);

static void keyboard (unsigned char key, int x, int y) {
    unsigned char esc = 27, space = ' ';
    if (key == esc) {
        exit(EXIT_SUCCESS);
    } else if (key == space) {
        ::running = !::running;
    } else if (key == 's') {
        ::running = !::running;
        idle();
        ::running = !::running;
    } else if (key == 'z') {
        zoom(true);
    } else if (key == 'x') {
        zoom(false);
    }
}

void display_physics (const vector<string> &args) {
    if (args.size() != 1 && args.size() != 2) {
        cout << "Runs the simulation with an OpenGL display." << endl;
        cout << "Arguments:" << endl;
        cout << "    <scene-file>: JSON file describing the simulation setup"
             << endl;
        cout << "    <out-dir> (optional): Directory to save output in" << endl;
        exit(EXIT_FAILURE);
    }
    string json_file = args[0];
    string outprefix = args.size()>1 ? args[1] : "";
cout << "init_physics" << endl;
    if (!outprefix.empty())
        ensure_existing_directory(outprefix);
    init_physics(json_file, outprefix, false);
    if (!outprefix.empty())
        save(sim, 0);
    GlutCallbacks cb;
    cb.idle = idle;
    cb.keyboard = keyboard;
    // cout << "???? 7\n";
    run_glut(cb);
}

void display_resume (const vector<string> &args) {
    if (args.size() != 2) {
        cout << "Resumes an incomplete simulation." << endl;
        cout << "Arguments:" << endl;
        cout << "    <out-dir>: Directory containing simulation output files"
             << endl;
        cout << "    <resume-frame>: Frame number to resume from" << endl;
        exit(EXIT_FAILURE);
    }
    init_resume(args);
    GlutCallbacks cb;
    cb.idle = idle;
    cb.keyboard = keyboard;
    run_glut(cb);
}

#else

void display_physics (const vector<string> &args) {opengl_fail();}

void display_resume (const vector<string> &args) {opengl_fail();}

#endif // NO_OPENGL
