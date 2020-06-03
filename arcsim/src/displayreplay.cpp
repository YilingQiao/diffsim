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

#include "displayreplay.hpp"

#include "conf.hpp"
#include "display.hpp"
#include "io.hpp"
#include "misc.hpp"
#include "opengl.hpp"
#include <cstdio>
#include <fstream>
using namespace std;

#ifndef NO_OPENGL

static string inprefix, outprefix;
static int frameskip;

static bool running = false;

static void reload () {
    int fullframe = ::frame*::frameskip;
    sim.time = fullframe * sim.frame_time;

    cout << stringf("%s/%04d_",inprefix.c_str(), fullframe) << endl;
    cout << stringf("%s/%04d_rig",inprefix.c_str(), fullframe) << endl;

    load_objs(sim.cloth_meshes, stringf("%s/%04d_",inprefix.c_str(), fullframe));
    load_objs(sim.obstacle_meshes, stringf("%s/%04d_rig",inprefix.c_str(), fullframe));

    
    // if (sim.cloth_meshes[0]->verts.empty()) {
    //     if (::frame == 0)
    //         exit(EXIT_FAILURE);
    //     if (!outprefix.empty())
    //         exit(EXIT_SUCCESS);
    //     ::frame = 0;
    //     reload();
    // }
    // for (int o = 0; o < sim.obstacles.size(); o++)
    //     sim.obstacles[o].get_mesh(sim.time);
}

static void idle () {
    if (!running)
        return;
    fps.tick();
    if (!outprefix.empty()) {
        char filename[256];
        snprintf(filename, 256, "%s/%04d.png", outprefix.c_str(), ::frame);
        save_screenshot(filename);
    }
    ::frame++;
    reload();
    fps.tock();
    redisplay();
}

static void keyboard (unsigned char key, int x, int y) {
    unsigned char esc = 27, space = ' ';
    if (key == esc) {
        exit(0);
    } else if (key == space) {
        running = !running;
    }
}

static void special (int key, int x, int y) {
    bool shift = glutGetModifiers() & GLUT_ACTIVE_SHIFT,
         alt = glutGetModifiers() & GLUT_ACTIVE_ALT;
    int delta = alt ? 100 : shift ? 10 : 1;
    if (key == GLUT_KEY_LEFT) {
        ::frame -= delta;
        reload();
    } else if (key == GLUT_KEY_RIGHT) {
        ::frame += delta;
        reload();
    } else if (key == GLUT_KEY_HOME) {
        ::frame = 0;
        reload();
    }
    redisplay();
}

void display_replay (const vector<string> &args) {
    if (args.size() < 1 || args.size() > 2) {
        cout << "Replays the results of a simulation." << endl;
        cout << "Arguments:" << endl;
        cout << "    <out-dir>: Directory containing simulation output files"
             << endl;
        cout << "    <sshot-dir> (optional): Directory to save images" << endl;
        exit(EXIT_FAILURE);
    }
    cout << "reply 1 " << endl;
    ::inprefix = args[0];
    ::outprefix = args.size()>1 ? args[1] : "";
    cout << "reply 2 " << endl;
    ::frameskip = 1;
    if (!::outprefix.empty())
        ensure_existing_directory(::outprefix);
    cout << "reply 3 " << endl;
    char config_backup_name[256];
    snprintf(config_backup_name, 256, "%s/%s", inprefix.c_str(), "conf.json");
    cout << "reply 4 " << endl;
    load_json(config_backup_name, sim);
    cout << "reply 5 " << endl;
    prepare(sim);
    cout << "reply 6 " << endl;
    reload();
    cout << "reply 7 " << endl;
    GlutCallbacks cb;
    cb.idle = idle;
    cb.keyboard = keyboard;
    cb.special = special;
    run_glut(cb);
    cout << "reply 8 " << endl;
}

#else

void display_replay (const vector<string> &args) {opengl_fail();}

#endif // NO_OPENGL
