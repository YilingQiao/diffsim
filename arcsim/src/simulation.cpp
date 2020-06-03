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

#include "simulation.hpp"

#include "collision.hpp"
#include "dynamicremesh.hpp"
#include "geometry.hpp"
#include "magic.hpp"
#include "nearobs.hpp"
#include "physics.hpp"
#include "plasticity.hpp"
#include "popfilter.hpp"
#include "proximity.hpp"
#include "separate.hpp"
#include "obstacle.hpp"
#include <iostream>
#include <fstream>
using namespace std;
using torch::Tensor;

static const bool verbose = false;
static const int proximity = Simulation::Proximity,
                 physics = Simulation::Physics,
                 strainlimiting = Simulation::StrainLimiting,
                 collision = Simulation::Collision,
                 remeshing = Simulation::Remeshing,
                 separation = Simulation::Separation,
                 popfilter = Simulation::PopFilter,
                 plasticity = Simulation::Plasticity;

void physics_step (Simulation &sim, const vector<Constraint*> &cons);
void plasticity_step (Simulation &sim);
void strainlimiting_step (Simulation &sim, const vector<Constraint*> &cons);
void strainzeroing_step (Simulation &sim);
void equilibration_step (Simulation &sim);
void collision_step (Simulation &sim);
void remeshing_step (Simulation &sim, bool initializing=false);

void validate_handles (const Simulation &sim);

void prepare (Simulation &sim) {
    sim.step = 0;
    sim.frame = 0;
    sim.cloth_meshes.resize(sim.cloths.size());
    for (int c = 0; c < sim.cloths.size(); c++) {
        compute_masses(sim.cloths[c]);
        sim.cloth_meshes[c] = &sim.cloths[c].mesh;
        update_x0(*sim.cloth_meshes[c]);
    }
    sim.obstacle_meshes.resize(sim.obstacles.size());
    for (int o = 0; o < sim.obstacles.size(); o++) {
        obs_compute_masses(sim.obstacles[o]);
        sim.obstacle_meshes[o] = &sim.obstacles[o].get_mesh();
        update_x0(*sim.obstacle_meshes[o]);
    }
}

void relax_initial_state (Simulation &sim) {
    validate_handles(sim);
    return;
    if (::magic.preserve_creases)
        for (int c = 0; c < sim.cloths.size(); c++)
            reset_plasticity(sim.cloths[c]);
    bool equilibrate = true;
    if (equilibrate) {
        equilibration_step(sim);
        remeshing_step(sim, true);
        equilibration_step(sim);
    } else {
        remeshing_step(sim, true);
        strainzeroing_step(sim);
        remeshing_step(sim, true);
        strainzeroing_step(sim);
    }
    if (::magic.preserve_creases)
        for (int c = 0; c < sim.cloths.size(); c++)
            reset_plasticity(sim.cloths[c]);
    ::magic.preserve_creases = false;
    if (::magic.fixed_high_res_mesh)
        sim.enabled[remeshing] = false;
}

void validate_handles (const Simulation &sim) {
    for (int h = 0; h < sim.handles.size(); h++) {
        vector<Node*> nodes = sim.handles[h]->get_nodes();
        for (int n = 0; n < nodes.size(); n++) {
            if (!nodes[n]->preserve) {
                cout << "Constrained node " << nodes[n]->index << " will not be preserved by remeshing" << endl;
                abort();
            }
        }
    }
}

vector<Constraint*> get_constraints (Simulation &sim, bool include_proximity);
void delete_constraints (const vector<Constraint*> &cons);
void update_obstacles (Simulation &sim, bool update_positions=true);

void advance_step (Simulation &sim);

void advance_frame (Simulation &sim) {
    for (int s = 0; s < sim.frame_steps; s++)
        advance_step(sim);
}

void advance_step (Simulation &sim) {

    Timer ti;
    ti.tick();
    sim.time = sim.time + sim.step_time;
    sim.step++;
    // cout << "\t\tstep=" << sim.step << endl;
    update_obstacles(sim, false);
    vector<Constraint*> cons = get_constraints(sim, true);
    physics_step(sim, cons);
    //plasticity_step(sim);
    //strainlimiting_step(sim, cons);
    collision_step(sim);
    if (sim.step % sim.frame_steps == 0) {
        remeshing_step(sim);
        sim.frame++;
        // cout << "\t\t\tframe="<<sim.frame<<endl;
    }
    delete_constraints(cons);
    ti.tock();
    // cout << "\t\ttime= "<< ti.last << endl;
}

vector<Constraint*> get_constraints (Simulation &sim, bool include_proximity) {
    // cout << "get_constraints" << endl;
    vector<Constraint*> cons;
    for (int h = 0; h < sim.handles.size(); h++)
        append(cons, sim.handles[h]->get_constraints(sim.time));
    //if (include_proximity && sim.enabled[proximity]) {
    //    sim.timers[proximity].tick();
    //    append(cons, proximity_constraints(sim.cloth_meshes,
    //                                       sim.obstacle_meshes,
    //                                       sim.friction, sim.obs_friction));
    //    sim.timers[proximity].tock();
    //}
    return cons;
}

void delete_constraints (const vector<Constraint*> &cons) {
    for (int c = 0; c < cons.size(); c++)
        delete cons[c];
}

// Steps

void update_velocities (vector<Mesh*> &meshes, vector<Tensor> &xold, Tensor dt);

void step_mesh (Mesh &mesh, Tensor dt);
void step_obstacle (Obstacle &obstacle, Tensor dt);

void physics_step (Simulation &sim, const vector<Constraint*> &cons) {
    // cout << "physics_step" << endl;
    if (!sim.enabled[physics])
        return;
    sim.timers[physics].tick();
    for (int c = 0; c < sim.cloths.size(); c++) {
        int nn = sim.cloths[c].mesh.nodes.size();
        Tensor fext = torch::zeros({nn,3}, TNOPT)*0;
        Tensor Jext = torch::zeros({nn,3,3}, TNOPT);
        
        add_external_forces(sim.cloths[c], sim.gravity, sim.wind, fext, Jext);
        
            
        for (int m = 0; m < sim.morphs.size(); m++)
            if (sim.morphs[m].mesh == &sim.cloths[c].mesh)
                add_morph_forces(sim.cloths[c], sim.morphs[m], sim.time,
                                 sim.step_time, fext, Jext);
        implicit_update(sim.cloths[c], fext, Jext, cons, sim.step_time, false);
    }

    
    for (int o = 0; o < sim.obstacles.size(); o++) {
        int nn = 2;
        Tensor fext = torch::zeros({nn,3}, TNOPT);
        Tensor Jext = torch::zeros({nn,3,3}, TNOPT);
        // just for test
        obs_add_external_forces(sim.obstacles[o], sim.gravity, sim.wind, fext, Jext);
        
        //fext = fext;
        //cout << "obs fext = " << fext << endl;
        obs_implicit_update(sim.obstacles[o], sim.obstacle_meshes, fext, Jext, cons, sim.step_time, false);
    }
    
    

    for (int c = 0; c < sim.cloth_meshes.size(); c++)
        step_mesh(*sim.cloth_meshes[c], sim.step_time);
    //for (int o = 0; o < sim.obstacle_meshes.size(); o++)
    //    step_mesh(*sim.obstacle_meshes[o], sim.step_time);
    for (int o = 0; o < sim.obstacles.size(); o++)
        step_obstacle(sim.obstacles[o], sim.step_time);
    sim.timers[physics].tock();
}

void step_mesh (Mesh &mesh, Tensor dt) {
    for (int n = 0; n < mesh.nodes.size(); n++) {
        mesh.nodes[n]->x = mesh.nodes[n]->x + mesh.nodes[n]->v*dt;
        mesh.nodes[n]->xold = mesh.nodes[n]->x;
    }
}

// void apply_transformation (Mesh& mesh, const Transformation& tr) {
//    apply_transformation_onto(mesh, mesh, tr);
//}


void step_obstacle (Obstacle &obstacle, Tensor dt) {
    Node *dummy_node = obstacle.curr_state_mesh.dummy_node;

    //cout << "velocity -------------" << endl;
    ////cout << dummy_node->x0 ;
    //cout << dummy_node->x ;

    if (dummy_node->movable)
        dummy_node->x    = dummy_node->v * dt + dummy_node->x;
    dummy_node->xold = dummy_node->x;


    //cout << dummy_node->x0 ;
    
    Tensor euler = dummy_node->x.slice(0, 0, 3);
    Tensor trans = dummy_node->x.slice(0, 3, 6);

    Transformation tr;
    tr.rotation    = Quaternion::from_euler(euler);
    tr.translation = trans;
    tr.scale       = dummy_node->scale;
    //cout << tr.rotation << endl;
    //cout << tr.translation << endl;
    //cout << tr.scale << endl;
    //cout << "step_obstacle" << endl;
    apply_transformation_onto(obstacle.base_mesh, obstacle.curr_state_mesh, tr);

    Mesh &mesh = obstacle.curr_state_mesh;

    for (int n = 0; n < mesh.nodes.size(); n++) {
        mesh.nodes[n]->xold = mesh.nodes[n]->x;
    }

}

void plasticity_step (Simulation &sim) {
    if (!sim.enabled[plasticity])
        return;
    // cout << "plasticity_step" << endl;
    sim.timers[plasticity].tick();
    for (int c = 0; c < sim.cloths.size(); c++) {
        plastic_update(sim.cloths[c]);
        optimize_plastic_embedding(sim.cloths[c]);
    }
    sim.timers[plasticity].tock();
}

void strainlimiting_step (Simulation &sim, const vector<Constraint*> &cons) {
    // cout << "strainlimiting_step" << endl;
}

void equilibration_step (Simulation &sim) {
    sim.timers[remeshing].tick();
    vector<Constraint*> cons;// = get_constraints(sim, true);
    // double stiff = 1;
    // swap(stiff, ::magic.handle_stiffness);
    for (int c = 0; c < sim.cloths.size(); c++) {
        Mesh &mesh = sim.cloths[c].mesh;
        for (int n = 0; n < mesh.nodes.size(); n++)
            mesh.nodes[n]->acceleration = ZERO3;
        apply_pop_filter(sim.cloths[c], cons, 1);
    }
    // swap(stiff, ::magic.handle_stiffness);
    sim.timers[remeshing].tock();
    delete_constraints(cons);
    cons = get_constraints(sim, false);
    if (sim.enabled[collision]) {
        sim.timers[collision].tick();
        collision_response(sim, sim.cloth_meshes, cons, sim.obstacle_meshes);
        sim.timers[collision].tock();
    }
    delete_constraints(cons);
}

void strainzeroing_step (Simulation &sim) {
}

void collision_step (Simulation &sim) {
    if (!sim.enabled[collision])
        return;
    // cout << "collision_step" << endl;
    sim.timers[collision].tick();
    vector<Tensor> xold = node_positions(sim.cloth_meshes);
    vector<Constraint*> cons = get_constraints(sim, false);
    collision_response(sim, sim.cloth_meshes, cons, sim.obstacle_meshes);
    delete_constraints(cons);
    update_velocities(sim.cloth_meshes, xold, sim.step_time);
    sim.timers[collision].tock();
}

void remeshing_step (Simulation &sim, bool initializing) {
    if (!sim.enabled[remeshing])
        return;
    // copy old meshes
    vector<Mesh> old_meshes(sim.cloths.size());
    vector<Mesh*> old_meshes_p(sim.cloths.size()); // for symmetry in separate()
    for (int c = 0; c < sim.cloths.size(); c++) {
        old_meshes[c] = deep_copy(sim.cloths[c].mesh);
        old_meshes_p[c] = &old_meshes[c];
    }
    // back up residuals
    typedef vector<Residual> MeshResidual;
    vector<MeshResidual> res;
    if (sim.enabled[plasticity] && !initializing) {
        sim.timers[plasticity].tick();
        res.resize(sim.cloths.size());
        for (int c = 0; c < sim.cloths.size(); c++)
            res[c] = back_up_residuals(sim.cloths[c].mesh);
        sim.timers[plasticity].tock();
    }
    // remesh
    sim.timers[remeshing].tick();
    for (int c = 0; c < sim.cloths.size(); c++) {
        if (::magic.fixed_high_res_mesh)
            static_remesh(sim.cloths[c]);
        else {
            vector<Plane> planes = nearest_obstacle_planes(sim.cloths[c].mesh,
                                                           sim.obstacle_meshes);
            dynamic_remesh(sim.cloths[c], planes, sim.enabled[plasticity]);
        }
    }
    sim.timers[remeshing].tock();
    // restore residuals
    if (sim.enabled[plasticity] && !initializing) {
        sim.timers[plasticity].tick();
        for (int c = 0; c < sim.cloths.size(); c++)
            restore_residuals(sim.cloths[c].mesh, old_meshes[c], res[c]);
        sim.timers[plasticity].tock();
    }
    // separate
    if (sim.enabled[separation]) {
        sim.timers[separation].tick();
        separate(sim.cloth_meshes, old_meshes_p, sim.obstacle_meshes);
        sim.timers[separation].tock();
    }
    // apply pop filter
    if (sim.enabled[popfilter] && !initializing) {
        sim.timers[popfilter].tick();
        vector<Constraint*> cons = get_constraints(sim, true);
        for (int c = 0; c < sim.cloths.size(); c++)
            apply_pop_filter(sim.cloths[c], cons);
        delete_constraints(cons);
        sim.timers[popfilter].tock();
    }
    // delete old meshes
    for (int c = 0; c < sim.cloths.size(); c++)
        delete_mesh(old_meshes[c]);
}

void update_velocities (vector<Mesh*> &meshes, vector<Tensor> &xold, Tensor dt) {
    Tensor inv_dt = 1/dt;
#pragma omp parallel for
    for (int n = 0; n < xold.size(); n++) {
        Node *node = get<Node>(n, meshes);
        node->v = node->v + (node->x - xold[n])*inv_dt;
    }
}

void update_obstacles (Simulation &sim, bool update_positions) {
    // cout << "update_obstacles" << endl;
    for (int o = 0; o < sim.obstacles.size(); o++) {
        sim.obstacles[o].get_mesh(sim.time);
        // sim.obstacles[o].blend_with_previous(sim.time, sim.step_time, blend);
        if (!update_positions) {
            // put positions back where they were
            Mesh &mesh = sim.obstacles[o].get_mesh();
            for (int n = 0; n < mesh.nodes.size(); n++) {
                Node *node = mesh.nodes[n];
                node->v = (node->x - node->x0)/sim.step_time;
                node->x = node->x0;
            }
        }
    }
}

// Helper functions

template <typename Prim> int size (const vector<Mesh*> &meshes) {
    int np = 0;
    for (int m = 0; m < meshes.size(); m++) np += get<Prim>(*meshes[m]).size();
    return np;
}
template int size<Vert> (const vector<Mesh*>&);
template int size<Node> (const vector<Mesh*>&);
template int size<Edge> (const vector<Mesh*>&);
template int size<Face> (const vector<Mesh*>&);

template <typename Prim> int get_index (const Prim *p,
                                        const vector<Mesh*> &meshes) {
    int i = 0;
    for (int m = 0; m < meshes.size(); m++) {
        const vector<Prim*> &ps = get<Prim>(*meshes[m]);
        if (p->index < ps.size() && p == ps[p->index])
            return i + p->index;
        else i += ps.size();
    }
    return -1;
}
template int get_index (const Vert*, const vector<Mesh*>&);
template int get_index (const Node*, const vector<Mesh*>&);
template int get_index (const Edge*, const vector<Mesh*>&);
template int get_index (const Face*, const vector<Mesh*>&);

template <typename Prim> Prim *get (int i, const vector<Mesh*> &meshes) {
    for (int m = 0; m < meshes.size(); m++) {
        const vector<Prim*> &ps = get<Prim>(*meshes[m]);
        if (i < ps.size())
            return ps[i];
        else
            i -= ps.size();
    }
    return NULL;
}
template Vert *get (int, const vector<Mesh*>&);
template Node *get (int, const vector<Mesh*>&);
template Edge *get (int, const vector<Mesh*>&);
template Face *get (int, const vector<Mesh*>&);

vector<Tensor> node_positions (const vector<Mesh*> &meshes) {
    vector<Tensor> xs(size<Node>(meshes));
    for (int n = 0; n < xs.size(); n++)
        xs[n] = get<Node>(n, meshes)->x;
    return xs;
}
