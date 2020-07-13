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

#include "conf.hpp"

#include "io.hpp"
#include "magic.hpp"
#include "mot_parser.hpp"
#include "util.hpp"
#include <cassert>
#include <cfloat>
#include <json/json.h>
#include <fstream>
// #include <png.h>
#include "sstream"
using namespace std;
using torch::Tensor;

void parse (bool&, const Json::Value&);
void parse (int&, const Json::Value&);
void parse (double&, const Json::Value&);
void parse (string&, const Json::Value&);

void complain (const Json::Value &json, const string &expected);

template <int n>
void parse_tn (Tensor &x, const Json::Value &json, const Tensor &x0=ZERO) {
    if (json.isNull()){
        if (n == 1)
            assert(x0.dim()==0);
        else
            assert(x0.dim()==1 && x0.size(0)==n);
        x = x0;
    }
    else {
        if (n == 1) {
            double tmp;
            parse(tmp, json);
            x = ONE*tmp;
            return;
        }
        if (!json.isArray()) complain(json, "array");
        assert(json.size() == n);
        vector<double> v(n);
        for (int i = 0; i < n; i++)
            parse(v[i], json[i]);
        x = torch::tensor(v, TNOPT);
    }
}

template <typename T> void parse (vector<T> &v, const Json::Value &json) {
    if (!json.isArray()) complain(json, "array");
    v.resize(json.size());
    for (int i = 0; i < json.size(); i++)
        parse(v[i], json[i]);
}

template <typename T> void parse (T &x, const Json::Value &json, const T &x0) {
    if (json.isNull())
        x = x0;
    else
        parse(x, json);
}

void parse (Cloth&, const Json::Value&);
void parse_motions (vector<Motion>&, const Json::Value&);
void parse_handles (vector<Handle*>&, const Json::Value&,
                    const vector<Cloth>&, const vector<Motion>&);
void parse_obstacles (vector<Obstacle>&, const Json::Value&,
                      const vector<Motion>&);
void parse_morphs (vector<Morph>&, const Json::Value&, const vector<Cloth> &);
void parse (Wind&, const Json::Value&);
void parse (Magic&, const Json::Value&);

void load_json (const string &configFilename, Simulation &sim) {
    Json::Value json;
    Json::Reader reader;
    ifstream file(configFilename.c_str());
    bool parsingSuccessful = reader.parse(file, json);
    if(!parsingSuccessful) {
        fprintf(stderr, "Error reading file: %s\n", configFilename.c_str());
        fprintf(stderr, "%s", reader.getFormatedErrorMessages().c_str());
        abort();
    }
    file.close();
    // Gather general data
    if (!json["frame_time"].empty()) {
        parse_tn<1>(sim.frame_time, json["frame_time"]);
        parse(sim.frame_steps, json["frame_steps"], 1);
        sim.step_time = sim.frame_time/sim.frame_steps;
        parse_tn<1>(sim.end_time, json["end_time"], infinity);
        parse(sim.end_frame, json["end_frame"], 2147483647);
    } else if (!json["timestep"].empty()) {
        parse_tn<1>(sim.step_time, json["timestep"]);
        parse(sim.frame_steps, json["save_frames"], 1);
        sim.frame_time = sim.step_time*sim.frame_steps;
        parse_tn<1>(sim.end_time, json["duration"], infinity);
        sim.end_frame = 2147483647;
    }
    sim.time = ZERO;
    parse(sim.cloths, json["cloths"]);
    parse_motions(sim.motions, json["motions"]);
    parse_handles(sim.handles, json["handles"], sim.cloths, sim.motions);
    parse_obstacles(sim.obstacles, json["obstacles"], sim.motions);
    //cout << "???? 3\n";
    parse_morphs(sim.morphs, json["morphs"], sim.cloths);
    parse_tn<3>(sim.gravity, json["gravity"], ZERO3);
    parse(sim.wind, json["wind"]);
    parse_tn<1>(sim.friction, json["friction"], 0.6*ONE);
    parse_tn<1>(sim.obs_friction, json["obs_friction"], 0.3*ONE);
    string module_names[] = {"proximity", "physics", "strainlimiting",
                             "collision", "remeshing", "separation",
                             "popfilter", "plasticity"};
    for (int i = 0; i < Simulation::nModules; i++) {
        sim.enabled[i] = true;
        for (int j = 0; j < json["disable"].size(); j++)
            if (json["disable"][j] == module_names[i])
                sim.enabled[i] = false;
    }
    parse(::magic, json["magic"]);
    // disable strain limiting and plasticity if not needed
    bool has_strain_limits = false, has_plasticity = false;
    for (int c = 0; c < sim.cloths.size(); c++)
        for (int m = 0; m < sim.cloths[c].materials.size(); m++) {
            Cloth::Material *mat = sim.cloths[c].materials[m];
            if (finite(mat->strain_min.item<double>()) || finite(mat->strain_max.item<double>()))
                has_strain_limits = true;
            if (finite(mat->yield_curv.item<double>()))
                has_plasticity = true;
        }
    if (!has_strain_limits)
        sim.enabled[Simulation::StrainLimiting] = false;
    if (!has_plasticity)
        sim.enabled[Simulation::Plasticity] = false;
    //cout << "???? 4\n";
}

// Basic data types

void complain (const Json::Value &json, const string &expected) {
    cout << "Expected " << expected << ", found " << json << " instead" << endl;
    abort();
}

void parse (bool &b, const Json::Value &json) {
    if (!json.isBool()) complain(json, "boolean");
    b = json.asBool();
}
void parse (int &n, const Json::Value &json) {
    if (!json.isIntegral()) complain(json, "integer");
    n = json.asInt();
}
void parse (double &x, const Json::Value &json) {
    if (!json.isNumeric()) complain(json, "real");
    x = json.asDouble();
}
void parse (string &s, const Json::Value &json) {
    if (!json.isString()) complain(json, "string");
    s = json.asString();
}

// Cloth

void parse (Transformation&, const Json::Value&);
void parse (Cloth::Material*&, const Json::Value&);
void parse (Cloth::Remeshing&, const Json::Value&);

struct Velocity {Tensor v, w; Tensor o;};
void parse (Velocity &, const Json::Value &);
void apply_velocity (Mesh &mesh, const Velocity &vel);
 
#ifndef FAST_MODE
void parse (Cloth &cloth, const Json::Value &json) {
    string filename;
    parse(filename, json["mesh"]);
    load_obj(cloth.mesh, filename);
    cloth.mesh.isCloth = true;
    Transformation transform;
    parse(transform, json["transform"]);
    if ((transform.scale != 1).item<int>())
        for (int v = 0; v < cloth.mesh.verts.size(); v++)
            cloth.mesh.verts[v]->u = cloth.mesh.verts[v]->u * transform.scale;
    compute_ms_data(cloth.mesh);
    apply_transformation(cloth.mesh, transform);
    //Velocity velocity;
    //parse(velocity, json["velocity"]);
    //apply_velocity(cloth.mesh, velocity);
    parse(cloth.materials, json["materials"]);
    parse(cloth.remeshing, json["remeshing"]);
}
#else
void parse (Cloth &cloth, const Json::Value &json) {
    string filename;
    parse(filename, json["mesh"]);
    load_obj(cloth.mesh, filename);


    Mesh &mesh = cloth.mesh;
    for (int n = 0; n < mesh.nodes.size(); n++) {
        Node *node = mesh.nodes[n];
        set_subvec(node->d_x, 0, node->x);
    }

    cloth.mesh.isCloth = true;
    Transformation transform;
    parse(transform, json["transform"]);
    if ((transform.scale != 1).item<int>())
        for (int v = 0; v < cloth.mesh.verts.size(); v++)
            cloth.mesh.verts[v]->u = cloth.mesh.verts[v]->u * transform.scale;
    compute_ms_data(cloth.mesh);
    apply_transformation(cloth.mesh, transform);
    //Velocity velocity;
    //parse(velocity, json["velocity"]);
    //apply_velocity(cloth.mesh, velocity);
    parse(cloth.materials, json["materials"]);
    parse(cloth.remeshing, json["remeshing"]);
}
#endif


void parse (Transformation& transform, const Json::Value &json) {
    vector<double> rot(4);
    parse_tn<3>(transform.translation, json["translate"], ZERO3);
    parse_tn<1>(transform.scale, json["scale"], ONE);
    parse(rot, json["rotate"], {0,0,0,0});
    transform.rotation = Quaternion::from_axisangle(
        torch::tensor({rot[1], rot[2], rot[3]},TNOPT), ONE*rot[0]*M_PI/180);
    transform.euler = Quaternion::to_euler(transform.rotation.s, transform.rotation.v);

}

void parse (Velocity &velocity, const Json::Value &json) {
    parse_tn<3>(velocity.v, json["linear"], ZERO3);
    parse_tn<3>(velocity.w, json["angular"], ZERO3);
    parse_tn<3>(velocity.o, json["origin"], ZERO3);
}

void apply_velocity (Mesh &mesh, const Velocity &vel) {
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->v = vel.v + cross(vel.w, mesh.nodes[n]->x - vel.o);
}

void load_material_data (Cloth::Material&, const string &filename, bool reuse);

void parse (Cloth::Material *&material, const Json::Value &json) {
    string filename;
    parse(filename, json["data"]);
    if (!material)
	material = new Cloth::Material;
    //memset(material, 0, sizeof(Cloth::Material));
    if(json["reuse"].isNull() || json["reuse"].asBool()==false)
        load_material_data(*material, filename, false);
    else
        load_material_data(*material, filename, true);
    double density_mult, stretching_mult, bending_mult, thicken;
    parse(density_mult, json["density_mult"], 1.);
    parse(stretching_mult, json["stretching_mult"], 1.);
    parse(bending_mult, json["bending_mult"], 1.);
    parse(thicken, json["thicken"], 1.);
    density_mult *= thicken;
    stretching_mult *= thicken;
    bending_mult *= thicken;
    material->density = material->density * density_mult;
   
    material->stretching = material->stretching * stretching_mult;
    material->bending = material->bending * bending_mult;
    // cout << material->bending << endl;
    parse_tn<1>(material->damping, json["damping"], ZERO);
    parse_tn<1>(material->strain_min, json["strain_limits"][0u], -infinity);
    parse_tn<1>(material->strain_max, json["strain_limits"][1], infinity);
    parse_tn<1>(material->yield_curv, json["yield_curv"], infinity);
    parse_tn<1>(material->weakening, json["weakening"], ZERO);
}

void parse (Cloth::Remeshing &remeshing, const Json::Value &json) {
    parse_tn<1>(remeshing.refine_angle, json["refine_angle"], infinity);
    parse_tn<1>(remeshing.refine_compression, json["refine_compression"], infinity);
    parse_tn<1>(remeshing.refine_velocity, json["refine_velocity"], infinity);
    parse_tn<1>(remeshing.size_min, json["size"][0u], -infinity);
    parse_tn<1>(remeshing.size_max, json["size"][1], infinity);
    parse_tn<1>(remeshing.aspect_min, json["aspect_min"], -infinity);
}

// Other things

void parse (Motion&, const Json::Value&);

void parse_motions (vector<Motion> &motions, const Json::Value &json) {
    if (json.isObject() && !json.isNull()) {
        string filename;
        double fps;
        Transformation trans;
        parse(filename, json["motfile"]);
        parse(fps, json["fps"]);
        parse(trans, json["transform"]);
        motions = load_mot(filename, fps);
        for (int m = 0; m < motions.size(); m++) {
            clean_up_quaternions(motions[m]);
            for (int p = 0; p < motions[m].points.size(); p++)
                motions[m].points[p].x = trans*motions[m].points[p].x;
            for (int p = 0; p < motions[m].points.size(); p++)
                fill_in_velocity(motions[m], p);
        }
    } else
        parse(motions, json);
}

void parse (Motion::Point&, const Json::Value&);

void parse (Motion &motion, const Json::Value &json) {
    parse(motion.points, json);
    for (int p = 0; p < motion.points.size(); p++)
        if ((motion.points[p].v.scale == infinity).item<int>()) // no velocity specified
            fill_in_velocity(motion, p);
}

void parse (Motion::Point &mp, const Json::Value &json) {
    parse_tn<1>(mp.t, json["time"]);
    parse(mp.x, json["transform"]);
    if (json["velocity"].isNull())
        mp.v.scale = infinity; // raise a flag
    else {
        parse(mp.v, json["velocity"]);
        if ((mp.v.scale == 1).item<int>())
            mp.v.scale = ZERO;
        if ((mp.v.rotation.s == 1).item<int>()) {
            mp.v.rotation.s = ZERO;
            mp.v.rotation.v = ZERO3;
        }
    }
}

void parse_handle (vector<Handle*> &, const Json::Value &,
                   const vector<Cloth> &, const vector<Motion> &);

void parse_handles (vector<Handle*> &hans, const Json::Value &jsons,
                    const vector<Cloth> &cloths, const vector<Motion> &motions){
    for (auto it : hans)
        delete it;
    hans.clear();
    for (int j = 0; j < jsons.size(); j++)
        parse_handle(hans, jsons[j], cloths, motions);
}

void parse_node_handle (vector<Handle*> &hans, const Json::Value &json,
                        const vector<Cloth> &cloths,
                        const vector<Motion> &motions);

void parse_circle_handle (vector<Handle*> &hans, const Json::Value &json,
                          const vector<Cloth> &cloths,
                          const vector<Motion> &motions);

void parse_glue_handle (vector<Handle*> &hans, const Json::Value &json,
                        const vector<Cloth> &cloths,
                        const vector<Motion> &motions);

void parse_handle (vector<Handle*> &hans, const Json::Value &json,
                   const vector<Cloth> &cloths, const vector<Motion> &motions) {
    string type;
    parse(type, json["type"], string("node"));
    int nhans = hans.size();
    if (type == "node")
        parse_node_handle(hans, json, cloths, motions);
    else if (type == "circle")
        parse_circle_handle(hans, json, cloths, motions);
    else if (type == "glue")
        parse_glue_handle(hans, json, cloths, motions);
    else {
        cout << "Unknown handle type " << type << endl;
        abort();
    }
    Tensor start_time, end_time, fade_time;
    parse_tn<1>(start_time, json["start_time"], ZERO);
    parse_tn<1>(end_time, json["end_time"], infinity);
    parse_tn<1>(fade_time, json["fade_time"], ZERO);
    for (int h = nhans; h < hans.size(); h++) {
        hans[h]->start_time = start_time;
        hans[h]->end_time = end_time;
        hans[h]->fade_time = fade_time;
    }
}

void parse_node_handle (vector<Handle*> &hans, const Json::Value &json,
                        const vector<Cloth> &cloths,
                        const vector<Motion> &motions) {
    int c, l, m;
    vector<int> ns;
    parse(c, json["cloth"], 0);
    parse(l, json["label"], -1);
    if (l == -1)
        parse(ns, json["nodes"]);
    parse(m, json["motion"], -1);
    const Mesh &mesh = cloths[c].mesh;
    const Motion *motion = (m != -1) ? &motions[m] : NULL;
    if (l != -1) {
        for (int n = 0; n < mesh.nodes.size(); n++) {
            if (mesh.nodes[n]->label != l)
                continue;
            NodeHandle *han = new NodeHandle;
            han->node = mesh.nodes[n];
            han->node->preserve = true;
            han->motion = motion;
            hans.push_back(han);
        }
    }
    if (!ns.empty()) {
        for (int i = 0; i < ns.size(); i++) {
            NodeHandle *han = new NodeHandle;
            han->node = mesh.nodes[ns[i]];
            han->node->preserve = true;
            han->motion = motion;
            hans.push_back(han);
        }
    }
}

void parse_circle_handle (vector<Handle*> &hans, const Json::Value &json,
                          const vector<Cloth> &cloths,
                          const vector<Motion> &motions) {
    CircleHandle *han = new CircleHandle;
    int c, m;
    parse(c, json["cloth"], 0);
    han->mesh = (Mesh*)&cloths[c].mesh;
    parse(han->label, json["label"]);
    parse(m, json["motion"], -1);
    han->motion = (m != -1) ? &motions[m] : NULL;
    parse_tn<1>(han->c, json["circumference"]);
    parse_tn<2>(han->u, json["u"]);
    parse_tn<3>(han->xc, json["center"]);
    parse_tn<3>(han->dx0, json["axis0"]);
    parse_tn<3>(han->dx1, json["axis1"]);
    hans.push_back(han);
}

void parse_glue_handle (vector<Handle*> &hans, const Json::Value &json,
                        const vector<Cloth> &cloths,
                        const vector<Motion> &motions) {
    GlueHandle *han = new GlueHandle;
    int c;
    vector<int> ns;
    parse(c, json["cloth"], 0);
    parse(ns, json["nodes"]);
    if (ns.size() != 2) {
        cout << "Must glue exactly two nodes together" << endl;
        abort();
    }
    const Mesh &mesh = cloths[c].mesh;
    han->nodes[0] = (Node*)mesh.nodes[ns[0]];
    han->nodes[1] = (Node*)mesh.nodes[ns[1]];
    hans.push_back(han);
}

void parse_obstacle (Obstacle&, const Json::Value&, const vector<Motion>&);

void parse_obstacles (vector<Obstacle> &obstacles, const Json::Value &json,
                      const vector<Motion> &motions) {
    if (json.isString()) {
        string fmt;
        parse(fmt, json);
        for (int i = 0; true; i++) {
            string filename = stringf(fmt, i);
            if (!fstream(filename.c_str(), ios::in))
                break;
            Obstacle obs;
            load_obj(obs.base_mesh, filename);
            obs.base_mesh.isCloth = false;
            obs.transform_spline = (i<motions.size()) ? &motions[i] : NULL;
            obs.start_time = ZERO;
            obs.end_time = infinity;
            obs.get_mesh(ZERO);
            obstacles.push_back(obs);
        }
    } else {
        for (int o = 0; o < obstacles.size(); o++) {
            delete_mesh(obstacles[o].curr_state_mesh) ;
            delete_mesh(obstacles[o].file_mesh) ;
            delete_mesh(obstacles[o].base_mesh) ;
        }
        obstacles.clear();

        obstacles.resize(json.size());
       
        for (int j = 0; j < json.size(); j++)
            parse_obstacle(obstacles[j], json[j], motions);
       
    }
}
#ifndef FAST_MODE
void parse_obstacle (Obstacle &obstacle, const Json::Value &json,
                     const vector<Motion> &motions) {
    string filename;
    parse(filename, json["mesh"]);
    load_obj(obstacle.base_mesh, filename);
    obstacle.base_mesh.isCloth = false;

    obstacle.file_mesh = deep_copy(obstacle.base_mesh);

    Transformation transform;
    parse(transform, json["transform"]); 
    apply_transformation(obstacle.base_mesh, transform);
    //obstacle.euler = transform.euler;
    int m;
    parse(m, json["motion"], -1);
    obstacle.transform_spline = (m != -1) ? &motions[m] : NULL;
    parse_tn<1>(obstacle.start_time, json["start_time"], ZERO);
    parse_tn<1>(obstacle.end_time, json["end_time"], infinity);

    obstacle.get_mesh(ZERO);

    Tensor com = ZERO3;
    int n_nodes = obstacle.base_mesh.nodes.size();
    
    for (int i = 0; i < n_nodes; i++) {
        com = com + obstacle.base_mesh.nodes[i]->x;
    }
    com = com / n_nodes;



    for (int i = 0; i < obstacle.base_mesh.nodes.size(); i++) {
        obstacle.base_mesh.nodes[i]->x = obstacle.base_mesh.nodes[i]->x - com;
    }

    obstacle.curr_state_mesh.dummy_node = new Node();
    Node *dummy_node = obstacle.curr_state_mesh.dummy_node;
    dummy_node->scale = ONE;
    dummy_node->x  = torch::cat({ZERO3, com});
    dummy_node->x0 = dummy_node->x + ZERO6;


    int movable;
    parse(movable, json["movable"], 1);
    dummy_node->movable = (movable==1) ? true : false;
    //cout << movable << " "  << dummy_node->movable << endl;

    parse_tn<6>(dummy_node->v, json["velocity"], ZERO6);

}
#else
void parse_obstacle (Obstacle &obstacle, const Json::Value &json,
                     const vector<Motion> &motions) {
    string filename;
    parse(filename, json["mesh"]);
    load_obj(obstacle.base_mesh, filename);
    obstacle.base_mesh.isCloth = false;

    obstacle.file_mesh = deep_copy(obstacle.base_mesh);

    Transformation transform;
    parse(transform, json["transform"]); 
    apply_transformation(obstacle.base_mesh, transform);
    //obstacle.euler = transform.euler;
    int m;
    parse(m, json["motion"], -1);
    obstacle.transform_spline = (m != -1) ? &motions[m] : NULL;
    parse_tn<1>(obstacle.start_time, json["start_time"], ZERO);
    parse_tn<1>(obstacle.end_time, json["end_time"], infinity);

    obstacle.get_mesh(ZERO);

    Tensor com = ZERO3;
    int n_nodes = obstacle.base_mesh.nodes.size();

    for (int i = 0; i < n_nodes; i++) {
        com = com + obstacle.base_mesh.nodes[i]->x;
    }
    com = com / n_nodes;

    for (int i = 0; i < obstacle.base_mesh.nodes.size(); i++) {
        obstacle.base_mesh.nodes[i]->x = obstacle.base_mesh.nodes[i]->x - com;
    
    }


    Mesh &base_mesh = obstacle.base_mesh;
    for (int n = 0; n < base_mesh.nodes.size(); n++) {
        Node *node = base_mesh.nodes[n];
        set_subvec(node->d_x, 0, node->x);
    }

    obstacle.curr_state_mesh.dummy_node = new Node();
    Node *dummy_node = obstacle.curr_state_mesh.dummy_node;
    dummy_node->scale = ONE;
    dummy_node->x  = torch::cat({ZERO3, com});
    dummy_node->x0 = dummy_node->x + ZERO6;


    int movable;
    parse(movable, json["movable"], 1);
    dummy_node->movable = (movable==1) ? true : false;
    //cout << movable << " "  << dummy_node->movable << endl;

    parse_tn<6>(dummy_node->v, json["velocity"], ZERO6);

}
#endif

void parse_morph (Morph&, const Json::Value&, const vector<Cloth>&);
void parse (Spline<Morph::Weights>::Point &, const Json::Value &);

void parse_morphs (vector<Morph> &morphs, const Json::Value &json,
                   const vector<Cloth> &cloths) {
    morphs.resize(json.size());
    for (int j = 0; j < json.size(); j++)
        parse_morph(morphs[j], json[j], cloths);
}

void parse_morph (Morph &morph, const Json::Value &json,
                  const vector<Cloth> &cloths) {
    int c;
    parse(c, json["cloth"], 0);
    morph.mesh = (Mesh*)&cloths[c].mesh;
    morph.targets.resize(json["targets"].size());
    for (int j = 0; j < json["targets"].size(); j++) {
        string filename;
        parse(filename, json["targets"][j]);
        load_obj(morph.targets[j], filename);
    }
    int nk = json["spline"].size();
    morph.weights.points.resize(nk);
    morph.log_stiffness.points.resize(nk);
    for (int k = 0; k < nk; k++) {
        const Json::Value &j = json["spline"][k];
        Tensor t; parse_tn<1>(t, j["time"]);
        morph.weights.points[k].t = morph.log_stiffness.points[k].t = t;
        int m; parse(m, j["target"]);
        morph.weights.points[k].x = torch::zeros({morph.targets.size()}, TNOPT);
        morph.weights.points[k].x[m] = 1;
        Tensor s; parse_tn<1>(s, j["stiffness"]);
        morph.log_stiffness.points[k].x = log(s);
    }
    for (int k = 0; k < nk; k++) {
        fill_in_velocity(morph.weights, k);
        fill_in_velocity(morph.log_stiffness, k);
    }
}

void parse (Wind &wind, const Json::Value &json) {
    parse_tn<1>(wind.density, json["density"], ONE);
    parse_tn<3>(wind.velocity, json["velocity"], ZERO3);
    parse_tn<1>(wind.drag, json["drag"], ZERO);
}

void parse (Magic &magic, const Json::Value &json) {
#define PARSE_MAGIC(param) parse_tn<1>(magic.param, json[#param], magic.param)
#define PARSE_MAGIC_BO(param) parse(magic.param, json[#param], magic.param)
    PARSE_MAGIC_BO(fixed_high_res_mesh);
    PARSE_MAGIC(handle_stiffness);
    PARSE_MAGIC(collision_stiffness);
    PARSE_MAGIC(repulsion_thickness);
    parse_tn<1>(magic.projection_thickness, json["projection_thickness"],
          0.1*magic.repulsion_thickness);
    PARSE_MAGIC(edge_flip_threshold);
    PARSE_MAGIC(rib_stiffening);
    PARSE_MAGIC_BO(combine_tensors);
    PARSE_MAGIC_BO(preserve_creases);
#undef PARSE_MAGIC
}

// JSON materials

void parse_stretching (StretchingSamples&, const Json::Value&, Cloth::Material&);
void parse_bending (BendingData&, const Json::Value&);

void load_material_data (Cloth::Material &material, const string &filename, bool reuse) {
    if (!reuse) {
    Json::Value json;
    Json::Reader reader;
    ifstream file(filename.c_str());
    bool parsingSuccessful = reader.parse(file, json);
    if(!parsingSuccessful) {
        fprintf(stderr, "Error reading file: %s\n", filename.c_str());
        fprintf(stderr, "%s", reader.getFormatedErrorMessages().c_str());
        abort();
    }
    file.close();
    parse_tn<1>(material.densityori, json["density"]);
    material.densityori.set_requires_grad(true);
    parse_stretching(material.stretching, json["stretching"], material);
    parse_bending(material.bendingori, json["bending"]);
    } else {
    cout << "reuse!!" << endl;
    }
    material.density = material.densityori*1;
    material.bending = material.bendingori.unsqueeze(0).unsqueeze(1);
    StretchingData data = torch::zeros({4,2,5},TNOPT);
    data.slice(1,0,1) = material.stretchingori[0].unsqueeze(1).unsqueeze(2).repeat({1,1,5});
    for (int i = 0; i < 5; i++) {
        data.slice(1,1,2).slice(2,i,i+1) = material.stretchingori[i+1].unsqueeze(1).unsqueeze(2);
    }
    data = data.unsqueeze(0);

    evaluate_stretching_samples(material.stretching, data);
}

void parse_stretching (StretchingSamples &samples, const Json::Value &json, Cloth::Material &mat) {
    Tensor st = torch::zeros({6,4}, TNOPT);
    Tensor tmp;
    for (int i = 0; i < 6; ++i) {
        parse_tn<4>(tmp, json[i]);
        st[i] = tmp;
    }
    mat.stretchingori = st.clone().detach().set_requires_grad(true);
}

void parse_bending (BendingData &data, const Json::Value &json) {
    Tensor be = torch::zeros({3,5},TNOPT);
    Tensor tmp;
    for (int i = 0; i < 3; i++) {
        parse_tn<5>(tmp, json[i]);
        be.slice(0,i,i+1) = tmp.unsqueeze(0);
    }
    data = be.clone().detach().set_requires_grad(true);
}
