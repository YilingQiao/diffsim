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

#include "display.hpp"

#ifndef NO_OPENGL

#include "bvh.hpp"
#include "geometry.hpp"
#include "io.hpp"
#include "timer.hpp"
#include "opengl.hpp"
#include "timer.hpp"
#include "util.hpp"
#include <sstream>
#include <vector>

using namespace std;
using torch::Tensor;

static vector<Mesh*> &meshes = sim.cloth_meshes;
string obj2png_filename;

extern int frame;
extern Timer fps;

struct View {
    double lat, lon;
    double off0, off1;
    double scale;
    View (): lat(0), lon(0), off0(0), off1(0), scale(0.5) {}
};

enum Pane {MaterialPane, PlasticPane, WorldPane};

bool pane_enabled[3] = {true, true, true};
int subwindows[3];
View views[3];

int get_pane () {return find(glutGetWindow(), subwindows, 3);}

void reshape (int w, int h) {
    int npanes = 0;
    for (int i = 0; i < 3; i++)
        if (pane_enabled[i])
            npanes++;
    int j = 0;
    for (int i = 0; i < 3; i++) {
        glutSetWindow(subwindows[i]);
        int x0 = w*j/npanes, x1 = pane_enabled[i] ? w*(j+1)/npanes : x0+1;
        glutPositionWindow(x0, 0);
        glutReshapeWindow(x1-x0, h);
        glViewport(0, 0, x1-x0, h);
        if (pane_enabled[i])
            j = j + 1;
    }
}

void vertex2 (const Tensor &x0) {
    double *x = x0.data<double>();
    glVertex2d(x[0], x[1]);
}

void vertex3 (const Tensor &x0) {
    double *x = x0.data<double>();
    glVertex3d(x[0], x[1], x[2]);
}

void normal3 (const Tensor &n0) {
    double *n = n0.data<double>();
    glNormal3d(n[0], n[1], n[2]);
}

void color (const Tensor &x0) {
    double *x = x0.data<double>();
    glColor3d(x[0], x[1], x[2]);
}

Tensor origami_color (const Face *face) {
    Tensor H = trace(face->S_plastic)/1000;
    return 0.9*torch::stack({1 + H, 1 - abs(H)/2, 1 - H});
}

void draw_mesh_ms (const Mesh &mesh, bool set_color=false) {
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < mesh.faces.size(); i++) {
        if (i % 256 == 0) {
            glEnd();
            glBegin(GL_TRIANGLES);
        }
        Face *face = mesh.faces[i];
        if (set_color)
            color(origami_color(face));
        for (int v = 0; v < 3; v++) {
            vertex2(face->v[v]->u);
        }
    }
    glEnd();
}

void draw_meshes_ms (bool set_color=false) {
    for (int m = 0; m < meshes.size(); m++)
        draw_mesh_ms(*meshes[m], set_color);
}

void shrink_face (const Face *face, double shrink_factor, double shrink_max,
                  Tensor u[3]) {
    Tensor u0 = face->v[0]->u, u1 = face->v[1]->u, u2 = face->v[2]->u;
    Tensor a = face->a;
    Tensor l = max(norm(u0 - u1), max(norm(u1 - u2), norm(u2 - u0)));
    Tensor h = 2*a/l;
    Tensor dh = min(h*shrink_factor, ONE*shrink_max);
    for (int v = 0; v < 3; v++) {
        Tensor e1 = normalize(face->v[NEXT(v)]->u - face->v[v]->u),
             e2 = normalize(face->v[PREV(v)]->u - face->v[v]->u);
        Tensor du = (e1 + e2)*dh / abs(wedge(e1, e2));
        u[v] = face->v[v]->u + du;
    }
}

void draw_meshes_ms_fancy () {
    double shrink_factor = 0.1, shrink_max = 0.5e-3;
    for (int m = 0; m < meshes.size(); m++) {
        const Mesh &mesh = *meshes[m];
        glBegin(GL_TRIANGLES);
        glColor3f(0.5,0.5,0.5);
        for (int f = 0; f < mesh.faces.size(); f++) {
            const Face *face = mesh.faces[f];
            for (int v = 0; v < 3; v++)
                vertex2(face->v[v]->u);
        }
        glEnd();
        glBegin(GL_TRIANGLES);
        for (int f = 0; f < mesh.faces.size(); f++) {
            const Face *face = mesh.faces[f];
            color(origami_color(face));
            glColor3f(0.9,0.9,0.9);
            Tensor u[3];
            shrink_face(face, shrink_factor, shrink_max, u);
            for (int v = 0; v < 3; v++)
                vertex2(u[v]);
        }
        glEnd();
    }
}

void draw_mesh_ps (const Mesh &mesh, bool set_color=false) {
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < mesh.faces.size(); i++) {
        Face *face = mesh.faces[i];
        if (i % 256 == 0) {
            glEnd();
            glBegin(GL_TRIANGLES);
        }
        normal3(nor<PS>(face));
        for (int v = 0; v < 3; v++)
            vertex3(face->v[v]->node->y);
    }
    glEnd();
}

void draw_meshes_ps (bool set_color=false) {
    for (int m = 0; m < meshes.size(); m++)
        draw_mesh_ps(*meshes[m], set_color);
}

template <Space s>
void draw_mesh (const Mesh &mesh, bool set_color=false) {
    if (set_color)
        glDisable(GL_COLOR_MATERIAL);
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < mesh.faces.size(); i++) {
        Face *face = mesh.faces[i];
        if (i % 256 == 0) {
            glEnd();
            glBegin(GL_TRIANGLES);
        }
        if (set_color) {
            int c = find((Mesh*)&mesh, ::meshes);
            static const double phi = (1+sqrt(5))/2;
            double hue = c*(2 - phi)*2*M_PI; // golden angle
            hue = -0.6*M_PI + hue; // looks better this way :/
            if (face->label % 2 == 1) hue += M_PI;
            static Tensor a = torch::tensor({0.92, -0.39, 0.},TNOPT), b = torch::tensor({0.05, 0.12, -0.99},TNOPT);
            Tensor frt0 = torch::tensor({0.7,0.7,0.7},TNOPT) + (a*cos(hue) + b*sin(hue))*0.3,
                 bak0 = frt0*0.5 + torch::tensor({0.5,0.5,0.5}, TNOPT);
            double *frt = frt0.data<double>(), *bak = bak0.data<double>();
            float front[4] = {frt[0], frt[1], frt[2], 1},
                  back[4] = {bak[0], bak[1], bak[2], 1};
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, front);
            glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, back);
            // color(area_color(face));
        }
        normal3(nor<s>(face));
        for (int v = 0; v < 3; v++)
            vertex3(pos<s>(face->v[v]->node));
    }
    glEnd();
    if (set_color)
        glEnable(GL_COLOR_MATERIAL);
}

template <Space s>
void draw_meshes (bool set_color=false) {
    for (int m = 0; m < meshes.size(); m++)
        draw_mesh<s>(*meshes[m], set_color);
}

template <Space s>
void draw_seam_or_boundary_edges () {
    glColor3f(0,0,0);
    glBegin(GL_LINES);
    for (int m = 0; m < meshes.size(); m++) {
        const Mesh &mesh = *meshes[m];
        for (int e = 0; e < mesh.edges.size(); e++) {
            const Edge *edge = mesh.edges[e];
            if (!is_seam_or_boundary(edge))
                continue;
            vertex3(pos<s>(edge->n[0]));
            vertex3(pos<s>(edge->n[1]));
        }
    }
    glEnd();
}

void draw_node_vels () {
    double dt = 0.01;
    glBegin(GL_LINES);
    for (int m = 0; m < meshes.size(); m++) {
        const Mesh &mesh = *meshes[m];
        for (int n = 0; n < mesh.nodes.size(); n++) {
            const Node *node = mesh.nodes[n];
            glColor3d(0,0,1);
            vertex3(node->x);
            vertex3(node->x + dt*node->v);
            glColor3d(1,0,0);
            vertex3(node->x);
            vertex3(node->x - dt*node->v);
        }
    }
    for (int o = 0; o < sim.obstacles.size(); o++) {
        const Mesh &mesh = sim.obstacles[o].get_mesh();
        for (int n = 0; n < mesh.nodes.size(); n++) {
            const Node *node = mesh.nodes[n];
            glColor3d(0,0,1);
            vertex3(node->x);
            vertex3(node->x + dt*node->v);
            glColor3d(1,0,0);
            vertex3(node->x);
            vertex3(node->x - dt*node->v);
        }
    }
    glEnd();
}

void draw_node_accels () {
    double dt2 = 1e-6;
    glBegin(GL_LINES);
    for (int m = 0; m < meshes.size(); m++) {
        const Mesh &mesh = *meshes[m];
        for (int n = 0; n < mesh.nodes.size(); n++) {
            const Node *node = mesh.nodes[n];
            glColor3d(0,0,1);
            vertex3(node->x);
            vertex3(node->x + dt2*node->acceleration);
            glColor3d(1,0,0);
            vertex3(node->x);
            vertex3(node->x - dt2*node->acceleration);
        }
    }
    glEnd();
}

void directional_light (int i, const vector<double> &dir, const vector<double> &dif) {
    float diffuse[4] = {dif[0], dif[1], dif[2], 1};
    float position[4] = {dir[0], dir[1], dir[2], 0};
    glEnable(GL_LIGHT0+i);
    glLightfv(GL_LIGHT0+i, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0+i, GL_POSITION, position);
}

void ambient_light (double a) {
    float ambient[4] = {a, a, a, 1};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
}

// void accumulate_ms_range (const Mesh &mesh, Vec2 &umin, Vec2 &umax) {
//     for (int f = 0; f < mesh.faces.size(); f++) {
//         for (int v = 0; v < 3; v++) {
//             Vec2 u = mesh.faces[f]->v[v]->u;
//             umin[0] = min(umin[0], u[0]);
//             umin[1] = min(umin[1], u[1]);
//             umax[0] = max(umax[0], u[0]);
//             umax[1] = max(umax[1], u[1]);
//         }
//     }
// }

// void fit_in_viewport (const Vec2 &umin, const Vec2 &umax, double aspect) {
//     Vec2 umid = (umin + umax)/2., usize = umax - umin;
//     if (usize[0] > usize[1]*aspect)
//         usize[1] = usize[0]/aspect;
//     else
//         usize[0] = usize[1]*aspect;
//     usize *= 1.1;
//     gluOrtho2D(umid[0] - usize[0]/2., umid[0] + usize[0]/2.,
//                umid[1] - usize[1]/2., umid[1] + usize[1]/2.);
// }

double aspect_ratio () {
    return (double)glutGet(GLUT_WINDOW_WIDTH)/glutGet(GLUT_WINDOW_HEIGHT);
}

void apply_view (const View &view, bool rotate=true) {
    glTranslatef(view.off0, view.off1, 0);
    glScalef(view.scale, view.scale, view.scale);
    if (rotate) {
        glRotatef(view.lat-90, 1,0,0);
        glRotatef(view.lon, 0,0,1);
    }
}

void display_material () {
    glClearColor(1,1,1,1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double a = aspect_ratio();
    gluOrtho2D(-a/2, a/2, -0.5, 0.5);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    apply_view(views[MaterialPane], false); // don't rotate material space
    // draw_meshes_ms_fancy();
    glColor3d(0.9,0.9,0.9);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    draw_meshes_ms(true);
    glColor4d(0,0,0, 0.2);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    draw_meshes_ms();
    // draw_ms_edges();
    glutSwapBuffers();
    GLenum errCode;
    const GLubyte *errString;
    if ((errCode = glGetError()) != GL_NO_ERROR) {
        errString = gluErrorString(errCode);
        cout << "OpenGL Error: " << errString << endl;
    }
}

void display_plastic () {
    glClearColor(1,1,1,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1,1);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, aspect_ratio(), 0.1, 10);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, -1);
    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);
    directional_light(0, {0,0,1}, {0.5,0.5,0.5});
    ambient_light(0.5);
    apply_view(views[PlasticPane]);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    draw_meshes<PS>(true);
    glColor4d(0,0,0, 0.2);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    draw_meshes<PS>();
    draw_seam_or_boundary_edges<PS>();
    glutSwapBuffers();
    GLenum errCode;
    const GLubyte *errString;
    if ((errCode = glGetError()) != GL_NO_ERROR) {
        errString = gluErrorString(errCode);
        cout << "OpenGL Error: " << errString << endl;
    }
}

void display_world () {
    glClearColor(1,1,1,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1,1);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, aspect_ratio(), 0.1, 10);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, -1);
    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);
    directional_light(0, {0,0,1}, {0.5,0.5,0.5});
    ambient_light(0.5);
    apply_view(views[WorldPane]);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    draw_meshes<WS>(true);
    glEnable(GL_CULL_FACE);
    glColor3f(0.8,0.8,0.8);
    for(int o = 0; o < sim.obstacles.size(); o++)
        draw_mesh<WS>(sim.obstacles[o].get_mesh());
    glDisable(GL_CULL_FACE);
    glColor4d(0,0,0, 0.2);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    draw_meshes<WS>();
    draw_seam_or_boundary_edges<WS>();
    glutSwapBuffers();
    GLenum errCode;
    const GLubyte *errString;
    if ((errCode = glGetError()) != GL_NO_ERROR) {
        errString = gluErrorString(errCode);
        cout << "OpenGL Error: " << errString << endl;
    }
}

struct MouseState {
    bool down;
    int x, y;
    enum {ROTATE, TRANSLATE, SCALE} func;
} mouse_state;

void zoom (bool in) {
    int pane = get_pane();
    if (pane == -1) {
        //ECHO("i don't know what to do with this event"); return;
        pane = 2;
    }
    View &view = views[pane];
    if (in)
        view.scale *= 1.2;
    else
        view.scale /= 1.2;
    glutPostRedisplay();
}

void mouse (int button, int state, int x, int y) {
    mouse_state.down = (state == GLUT_DOWN);
    mouse_state.x = x;
    mouse_state.y = y;
    int pane = get_pane();
    if (pane == -1) {ECHO("i don't know what to do with this event"); return;}
    View &view = views[pane];
    if ((button == 3) || (button == 4)) { // It's a wheel event
        mouse_state.func = MouseState::SCALE;
        // Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
        if (state == GLUT_UP) return; // Disregard redundant GLUT_UP events
        if (button == 3) {
            view.scale *= 1.2;
        } else {
            view.scale /= 1.2;
        }
        glutPostRedisplay();
    } else if (button == GLUT_LEFT_BUTTON) {
        mouse_state.func = MouseState::ROTATE;
    } else if (button == GLUT_MIDDLE_BUTTON) {
        mouse_state.func = MouseState::TRANSLATE;
    }
}

void motion (int x, int y) {
    if (!mouse_state.down)
        return;
    int pane = get_pane();
    if (pane == -1) {ECHO("i don't know what to do with this event"); return;}
    View &view = views[pane];
    if (mouse_state.func == MouseState::ROTATE) {
        double speed = 0.25;
        view.lon += (x - mouse_state.x)*speed;
        view.lat += (y - mouse_state.y)*speed;
        view.lat = clamp(view.lat, -90., 90.);
    } else if (mouse_state.func == MouseState::TRANSLATE) {
        double speed = 1e-3;
        view.off0 += (x - mouse_state.x)*speed;
        view.off1 -= (y - mouse_state.y)*speed;
    }
    mouse_state.x = x;
    mouse_state.y = y;
    glutPostRedisplay();
}

void nop () {} // apparently needed by GLUT 3.0

void run_glut (const GlutCallbacks &cb) {
    int argc = 1;
    char argv0[] = "";
    char *argv = argv0;
    glutInit(&argc, &argv);
    //cout << "???? 71\n";
    glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH|GLUT_MULTISAMPLE);
    glutInitWindowSize(1280,720);
    int window = glutCreateWindow("ARCSim");
    glutDisplayFunc(nop);
    //cout << "???? 72\n";
    glutReshapeFunc(reshape);
    glutIdleFunc(cb.idle);
    glutKeyboardFunc(cb.keyboard);
    glutSpecialFunc(cb.special);
    //cout << "???? 72\n";
    double x[4] = {0, 1280/3, 1280*2/3, 1280.};
    void (*display[3]) () = {display_material, display_plastic, display_world};
    for (int i = 0; i < 3; i++) {
        subwindows[i] = glutCreateSubWindow(window, x[i],0, x[i+1],720);
        glutDisplayFunc(display[i]);
        glutKeyboardFunc(cb.keyboard);
    //cout << "???? 73\n";
        glutSpecialFunc(cb.special);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
    //cout << "???? 74\n";
    }
    ::pane_enabled[PlasticPane] = sim.enabled[Simulation::Plasticity];
    //cout << "???? 75\n";
    glutMainLoop();
}

void redisplay () {
    for (int i = 0; i < 3; i++) {
        glutSetWindow(subwindows[i]);
        glutPostRedisplay();
    }
}

#endif // NO_OPENGL
